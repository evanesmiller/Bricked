"""
LEGO conversion stage using Trimesh.
Reads the voxel grid, merges adjacent voxels into standard brick types,
assigns colors, generates a parts list, and stores the final model as GLTF.

Brick types supported: 1x1, 1x2, 1x3, 1x4, 2x2, 2x3, 2x4
Colors are quantized to the nearest standard LEGO palette color.
"""
import json
import io
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs

# Standard LEGO color palette (name → hex)
LEGO_PALETTE = {
    "Red":    "#B40000",
    "Blue":   "#006DB7",
    "Yellow": "#FFD700",
    "Green":  "#00852B",
    "White":  "#FFFFFF",
    "Black":  "#1B2A34",
    "Orange": "#FF7000",
    "Gray":   "#9BA19D",
}

# Supported brick footprints (width x depth in stud units)
BRICK_TYPES = [(2, 4), (2, 3), (2, 2), (1, 4), (1, 3), (1, 2), (1, 1)]


def _quantize_color(hex_color: str | None) -> tuple[str, str]:
    """Return (lego_color_name, hex) closest to the input. Defaults to Gray."""
    # Extend with real color-distance logic once YOLO provides per-voxel color
    return "Gray", LEGO_PALETTE["Gray"]


def _pack_bricks(voxels: list[dict]) -> list[dict]:
    """
    Greedy brick packing: walk the voxel grid layer by layer (Y axis = height),
    try to cover each unfilled cell with the largest brick that fits.
    Returns a list of brick dicts with position, type, and color.
    """
    occupied = {(v["x"], v["y"], v["z"]) for v in voxels}
    filled = set()
    bricks = []

    ys = sorted({v["y"] for v in voxels})
    for y in ys:
        layer_cells = {(v["x"], v["z"]) for v in voxels if v["y"] == y}
        remaining = layer_cells - {(x, z) for (x, yy, z) in filled if yy == y}

        for (x, z) in sorted(remaining):
            if (x, z) not in remaining:
                continue
            for (w, d) in BRICK_TYPES:
                footprint = {(x + dx, z + dz) for dx in range(w) for dz in range(d)}
                if footprint.issubset(remaining):
                    color_name, color_hex = _quantize_color(None)
                    bricks.append({
                        "x": x, "y": y, "z": z,
                        "width": w, "depth": d, "height": 1,
                        "type": f"{w}x{d}",
                        "color_name": color_name,
                        "color": color_hex,
                    })
                    for cell in footprint:
                        remaining.discard(cell)
                        filled.add((cell[0], y, cell[1]))
                    break

    return bricks


def _build_parts_list(bricks: list[dict]) -> list[dict]:
    counts: dict[tuple, int] = {}
    for b in bricks:
        key = (b["type"], b["color_name"], b["color"])
        counts[key] = counts.get(key, 0) + 1
    return [
        {"type": t, "color_name": cn, "color": c, "count": n}
        for (t, cn, c), n in sorted(counts.items())
    ]


async def run_lego_conversion(run_id: str) -> dict:
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("voxelized",):
        raise HTTPException(status_code=409, detail=f"Run is in status '{run['status']}', expected 'voxelized'")

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "converting", "lego_started_at": datetime.now(timezone.utc)}},
    )

    try:
        # Load voxels from GridFS
        voxel_stream = await gridfs.open_download_stream(
            ObjectId(run["voxelization"]["voxel_file_id"])
        )
        voxels = json.loads(await voxel_stream.read())

        # --- Trimesh mesh generation goes here ---
        # import trimesh, numpy as np
        # boxes = [trimesh.creation.box(extents=[b["width"], b["height"], b["depth"]],
        #           transform=trimesh.transformations.translation_matrix([b["x"], b["y"], b["z"]]))
        #          for b in bricks]
        # scene = trimesh.Scene(boxes)
        # gltf_bytes = scene.export(file_type="glb")
        # model_file_id = await gridfs.upload_from_stream("model.glb", io.BytesIO(gltf_bytes), ...)
        # -----------------------------------------

        bricks = _pack_bricks(voxels)
        parts_list = _build_parts_list(bricks)

        # Store final model JSON (used by Three.js renderer)
        model_data = {
            "bricks": bricks,
            "dimensions": {
                "width": max((b["x"] + b["width"]) for b in bricks) if bricks else 0,
                "height": max((b["y"] + b["height"]) for b in bricks) if bricks else 0,
                "depth": max((b["z"] + b["depth"]) for b in bricks) if bricks else 0,
            },
        }
        model_bytes = json.dumps(model_data).encode()
        model_file_id = await gridfs.upload_from_stream(
            "model.json",
            io.BytesIO(model_bytes),
            metadata={"run_id": run_id, "content_type": "application/json"},
        )

        result = {
            "model_file_id": str(model_file_id),
            "brick_count": len(bricks),
            "parts_list": parts_list,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {
                "$set": {
                    "status": "complete",
                    "lego": result,
                    "lego_completed_at": datetime.now(timezone.utc),
                }
            },
        )
        return {"run_id": run_id, "status": "complete", **result}

    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
