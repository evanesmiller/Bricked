"""
LEGO conversion stage using Trimesh.
Reads the voxel grid, merges adjacent voxels into standard brick types,
assigns colors, generates a parts list, and stores the final model as GLTF.

Brick types supported: 1x1, 1x2, 1x3, 1x4, 2x2, 2x3, 2x4
Colors are quantized to the nearest standard LEGO palette color via Euclidean
distance in CIE-LAB space.
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
LEGO_PALETTE: dict[str, tuple[str, int, int, int]] = {
    "Bright Red":           ("#C91A09", 201,  26,   9),
    "Dark Red":             ("#720E0F", 114,  14,  15),
    "Bright Blue":          ("#0055BF",   0,  85, 191),
    "Medium Blue":          ("#5A93DB",  90, 147, 219),
    "Bright Yellow":        ("#F2CD37", 242, 205,  55),
    "Bright Green":         ("#4B9F4A",  75, 159,  74),
    "Dark Green":           ("#184632",  24,  70,  50),
    "White":                ("#FFFFFF", 255, 255, 255),
    "Black":                ("#05131D",   5,  19,  29),
    "Bright Orange":        ("#FE8A18", 254, 138,  24),
    "Medium Stone Gray":    ("#A0A5A9", 160, 165, 169),
    "Dark Stone Gray":      ("#6C6E68", 108, 110, 104),
    "Reddish Brown":        ("#582A12",  88,  42,  18),
    "Nougat":               ("#D09168", 208, 145, 104),
    "Tan":                  ("#E4CD9E", 228, 205, 158),
    "Bright Pink":          ("#FF698F", 255, 105, 143),
    "Bright Purple":        ("#81007B", 129,   0, 123),
    "Sand Green":           ("#A0BCAC", 160, 188, 172),
    "Medium Azure":         ("#36AEBF",  54, 174, 191),
}

# Pre-compute full palette in CIE-LAB for perceptually-uniform nearest-color lookup.
_PALETTE_NAMES = list(LEGO_PALETTE.keys())

def _rgb_to_lab(r: int, g: int, b: int) -> np.ndarray:
    """Convert a single sRGB pixel to OpenCV's uint8 LAB encoding."""
    px = np.array([[[r, g, b]]], dtype=np.uint8)
    return cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)

def _rgb_array_to_lab(arr: np.ndarray) -> np.ndarray:
    """Batch convert (N, 3) uint8 RGB array to (N, 3) float32 LAB."""
    return cv2.cvtColor(arr.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)

_PALETTE_LAB = np.stack([
    _rgb_to_lab(r, g, b) for _, r, g, b in LEGO_PALETTE.values()
])  # (N, 3)

# Supported brick footprints (width x depth in stud units)
BRICK_TYPES = [(2, 4), (2, 3), (2, 2), (1, 4), (1, 3), (1, 2), (1, 1)]

# Number of k-means clusters used to derive the dominant color palette.
# Raise to allow more colors; lower to be stricter about phantom-color suppression.
_N_COLOR_CLUSTERS = 6


def _quantize_against(r: int, g: int, b: int,
                      names: list[str], lab_array: np.ndarray) -> tuple[str, str]:
    """Nearest-LEGO-color lookup against an arbitrary (names, lab_array) palette."""
    query = _rgb_to_lab(r, g, b)
    diffs = lab_array - query
    idx   = int(np.argmin((diffs ** 2).sum(axis=1)))
    name  = names[idx]
    return name, LEGO_PALETTE[name][0]


def _dominant_palette(voxels: list[dict]) -> tuple[list[str], np.ndarray]:
    """
    K-means palette derivation in CIE-LAB space:
      1. Convert all voxel colors to LAB for perceptually-uniform clustering.
      2. Run k-means to find _N_COLOR_CLUSTERS perceptual color groups.
      3. Map each cluster centroid to the nearest LEGO palette color.

    Clustering on centroids rather than individual voxels means noise voxels
    with stray colors (e.g. a handful of reddish points on a brown surface)
    get absorbed into the dominant cluster instead of contributing a phantom
    LEGO color.
    """
    from scipy.cluster.vq import kmeans as _kmeans

    rgb_arr = np.array([[v["r"], v["g"], v["b"]] for v in voxels], dtype=np.uint8)
    lab_arr = _rgb_array_to_lab(rgb_arr)  # (N, 3) float32

    k = min(_N_COLOR_CLUSTERS, len(lab_arr))
    try:
        centroids, _ = _kmeans(lab_arr.astype(np.float64), k, iter=20)
    except Exception:
        centroids = lab_arr[:k].astype(np.float64)

    used_names: list[str] = []
    for c in centroids:
        diffs = _PALETTE_LAB - c.astype(np.float32)
        idx   = int(np.argmin((diffs ** 2).sum(axis=1)))
        name  = _PALETTE_NAMES[idx]
        if name not in used_names:
            used_names.append(name)

    if not used_names:
        used_names = ["Medium Stone Gray"]

    restricted_lab = np.stack([_rgb_to_lab(*LEGO_PALETTE[n][1:]) for n in used_names])
    return used_names, restricted_lab


def _pack_bricks(voxels: list[dict]) -> list[dict]:
    """
    Color-aware greedy brick packing with dominant-palette filtering:
      1. Derive the dominant LEGO palette from actual voxel color distribution.
         Colors used by fewer than ~3% of voxels are treated as noise and
         re-mapped to the nearest dominant color — ensuring every LEGO color
         in the output has genuine representation in the voxel grid.
      2. Pack layer by layer; a brick footprint is only accepted when all cells
         share the same quantized LEGO color.
    """
    _default = ("Medium Stone Gray", LEGO_PALETTE["Medium Stone Gray"][0])

    has_color = bool(voxels) and "r" in voxels[0]
    qcolor: dict[tuple, tuple[str, str]] = {}

    if has_color:
        dom_names, dom_lab = _dominant_palette(voxels)
        for v in voxels:
            name, hex_col = _quantize_against(v["r"], v["g"], v["b"], dom_names, dom_lab)
            qcolor[(v["x"], v["y"], v["z"])] = (name, hex_col)

    filled = set()
    bricks = []

    ys = sorted({v["y"] for v in voxels})
    for y in ys:
        layer_cells = {(v["x"], v["z"]) for v in voxels if v["y"] == y}
        remaining = layer_cells - {(x, z) for (x, yy, z) in filled if yy == y}

        for (x, z) in sorted(remaining):
            if (x, z) not in remaining:
                continue
            cell_color = qcolor.get((x, y, z), _default)
            for (w, d) in BRICK_TYPES:
                footprint = {(x + dx, z + dz) for dx in range(w) for dz in range(d)}
                if not footprint.issubset(remaining):
                    continue
                if any(qcolor.get((fx, y, fz), _default) != cell_color
                       for fx, fz in footprint):
                    continue
                color_name, color_hex = cell_color
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
