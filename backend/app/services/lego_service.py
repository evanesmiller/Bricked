"""
LEGO conversion stage using Trimesh.
Reads the voxel grid, merges adjacent voxels into standard brick types,
assigns colors, generates a parts list, and stores the final model as GLTF.

Brick types supported: 1x1, 1x2, 1x3, 1x4, 2x2, 2x3, 2x4
Colors are quantized to the nearest standard LEGO palette color via Euclidean
distance in RGB space.
"""
import cv2
import json
import io
import numpy as np
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs

# Standard LEGO color palette (name → (hex, R, G, B))
# 16 classic LEGO colors. Includes Bright Pink so warm flesh-to-magenta tones
# map correctly instead of falling through to Tan or Red.
LEGO_PALETTE: dict[str, tuple[str, int, int, int]] = {
    "Bright Red":       ("#C91A09", 201,  26,   9),
    "Bright Blue":      ("#0055BF",   0,  85, 191),
    "Bright Yellow":    ("#F2CD37", 242, 205,  55),
    "Bright Green":     ("#4B9F4A",  75, 159,  74),
    "Dark Green":       ("#184632",  24,  70,  50),
    "White":            ("#FFFFFF", 255, 255, 255),
    "Black":            ("#05131D",   5,  19,  29),
    "Bright Orange":    ("#FE8A18", 254, 138,  24),
    "Medium Stone Gray":("#A0A5A9", 160, 165, 169),
    "Dark Stone Gray":  ("#6C6E68", 108, 110, 104),
    "Reddish Brown":    ("#582A12",  88,  42,  18),
    "Tan":              ("#E4CD9E", 228, 205, 158),
    "Bright Pink":      ("#FF698F", 255, 105, 143),
    "Bright Purple":    ("#81007B", 129,   0, 123),
    "Sand Green":       ("#A0BCAC", 160, 188, 172),
    "Medium Azure":     ("#36AEBF",  54, 174, 191),
}

# Pre-compute palette in CIE-LAB for perceptually-uniform nearest-color lookup.
# LAB separates lightness (L) from chroma (a, b), so White vs Tan vs Gray stay
# far apart even though their RGB values can be misleadingly close.
_PALETTE_NAMES = list(LEGO_PALETTE.keys())

def _rgb_to_lab(r: int, g: int, b: int) -> np.ndarray:
    """Convert a single sRGB pixel to OpenCV's uint8 LAB encoding."""
    px = np.array([[[r, g, b]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

_PALETTE_LAB = np.stack([
    _rgb_to_lab(r, g, b) for _, r, g, b in LEGO_PALETTE.values()
])  # (16, 3)

# Supported brick footprints (width x depth in stud units)
BRICK_TYPES = [(2, 4), (2, 3), (2, 2), (1, 4), (1, 3), (1, 2), (1, 1)]


def _quantize_color(r: int, g: int, b: int) -> tuple[str, str]:
    """Return (lego_color_name, hex) nearest to (r, g, b) in CIE-LAB space."""
    query = _rgb_to_lab(r, g, b)
    diffs = _PALETTE_LAB - query          # (16, 3)
    idx   = int(np.argmin((diffs ** 2).sum(axis=1)))
    name  = _PALETTE_NAMES[idx]
    return name, LEGO_PALETTE[name][0]


def _pack_bricks(voxels: list[dict]) -> list[dict]:
    """
    Greedy brick packing: walk the voxel grid layer by layer (Y axis = height),
    try to cover each unfilled cell with the largest brick that fits.

    Per-brick color is determined by averaging the RGB of all voxels in the
    brick's footprint, then quantizing to the nearest LEGO palette color.
    """
    # Build a lookup from (x, y, z) → (r, g, b); fall back to gray if absent
    color_lookup: dict[tuple, tuple[int, int, int]] = {}
    has_color = "r" in voxels[0] if voxels else False
    if has_color:
        for v in voxels:
            color_lookup[(v["x"], v["y"], v["z"])] = (v["r"], v["g"], v["b"])

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
                    if has_color:
                        rgbs = [color_lookup.get((fx, y, fz), (128, 128, 128))
                                for fx, fz in footprint]
                        avg_r = int(round(sum(c[0] for c in rgbs) / len(rgbs)))
                        avg_g = int(round(sum(c[1] for c in rgbs) / len(rgbs)))
                        avg_b = int(round(sum(c[2] for c in rgbs) / len(rgbs)))
                        color_name, color_hex = _quantize_color(avg_r, avg_g, avg_b)
                    else:
                        color_name, color_hex = "Medium Stone Gray", LEGO_PALETTE["Medium Stone Gray"][0]

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
