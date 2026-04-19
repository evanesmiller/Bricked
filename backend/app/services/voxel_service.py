"""
Voxelization stage using Open3D + scipy.

Pipeline:
  1. Load reconstructed point cloud from GridFS
  2. Statistical outlier removal — drops isolated satellite clusters
  3. Open3D VoxelGrid at VOXEL_SIZE — converts the continuous cloud to a coarse
     discrete grid sized to produce ~300-500 LEGO bricks for a typical object
  4. Gaussian smoothing of the 3-D occupancy grid — replaces jagged/complex
     surface detail with smooth, simplified geometry:
       · 1-voxel protrusions fall below the threshold and are removed
       · shallow concavities are filled (surrounded voxels stay above threshold)
       · sharp corners round into smooth curves
  5. Morphological opening (erode → dilate) — removes any remaining silhouette
     fins or thin wings that survive the Gaussian pass
  6. Largest-connected-component filter — drops pieces detached by the opening
  7. Store cleaned voxel list to GridFS
"""
import asyncio
import io
import json
import logging
from datetime import datetime, timezone

import numpy as np
from bson import ObjectId
from fastapi import HTTPException
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    gaussian_filter,
    generate_binary_structure,
    label as nd_label,
)

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

# ── Simplification parameters ─────────────────────────────────────────────────
#
# VOXEL_SIZE controls the coarseness of the output grid.
# At 0.12 in [-1,1]³ the grid is ~17 cells per axis.  A solid cylindrical
# object filling that space yields ~1 500-2 000 voxels → ~300-500 LEGO bricks
# after greedy packing (average ~4 studs per brick).
#
# GAUSS_SIGMA of 1.0 smooths at the scale of ~1 voxel.  Combined with a
# threshold slightly below 0.5 (GAUSS_THRESH=0.40) the filter:
#   · removes 1-voxel bumps / small surface detail  (peak < 0.40 after blur)
#   · fills shallow concavities                     (≥ 5/6 neighbors filled → > 0.40)
#   · rounds sharp edges into smooth curves         (gradual falloff across corners)
#
# OPEN_ITERS=1 removes thin silhouette fins that the Gaussian pass may leave.

VOXEL_SIZE   = 0.07   # world units in [-1, 1]³  →  ~29 voxels per axis
GAUSS_SIGMA  = 0.5    # smoothing radius in voxels
GAUSS_THRESH = 0.45   # re-binarisation threshold after smoothing
OPEN_ITERS   = 1      # morphological opening iterations (fin removal)


def _build_voxel_grid(point_list: list[dict]) -> list[dict]:
    """
    Convert a point cloud (list of {x,y,z} dicts in [-1,1]³) to a simplified,
    LEGO-ready voxel grid.  Returns integer grid-index dicts {x, y, z}.
    """
    import open3d as o3d

    if not point_list:
        return []

    pts   = np.array([[p["x"], p["y"], p["z"]] for p in point_list], dtype=np.float64)
    n_raw = len(pts)

    # ── 1. Statistical outlier removal ───────────────────────────────────────
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    logger.debug("SOR: %d → %d points", n_raw, len(pcd.points))

    # ── 2. Coarse voxelisation ────────────────────────────────────────────────
    vg         = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
    raw_voxels = vg.get_voxels()
    if not raw_voxels:
        return []

    # ── 3. Build 3-D occupancy grid ───────────────────────────────────────────
    indices = np.array([v.grid_index for v in raw_voxels], dtype=np.int32)
    # +2 padding so the Gaussian and erosion kernels don't clip grid edges
    shape   = tuple(indices.max(axis=0) + 2)
    grid    = np.zeros(shape, dtype=np.float32)
    grid[indices[:, 0], indices[:, 1], indices[:, 2]] = 1.0
    n_pre   = int((grid > 0).sum())

    # ── 4. Gaussian smoothing — shape simplification ──────────────────────────
    smoothed = gaussian_filter(grid, sigma=GAUSS_SIGMA)
    grid_bin = smoothed >= GAUSS_THRESH

    logger.debug(
        "Gaussian (σ=%.1f, thresh=%.2f): %d → %d voxels",
        GAUSS_SIGMA, GAUSS_THRESH, n_pre, int(grid_bin.sum()),
    )

    # ── 5. Morphological opening — remove remaining thin fins ─────────────────
    struct  = generate_binary_structure(3, 1)   # 6-connected face kernel
    eroded  = binary_erosion(grid_bin, structure=struct, iterations=OPEN_ITERS, border_value=0)
    cleaned = binary_dilation(eroded,  structure=struct, iterations=OPEN_ITERS)

    # ── 6. Largest connected component ────────────────────────────────────────
    labeled, n_comp = nd_label(cleaned)
    if n_comp == 0:
        # Simplification removed everything (object too thin) — fall back
        logger.warning("Simplified grid is empty; falling back to raw grid")
        labeled, n_comp = nd_label(grid_bin if grid_bin.any() else (grid > 0))
    if n_comp == 0:
        return []

    sizes         = np.bincount(labeled.ravel())[1:]
    largest_label = int(np.argmax(sizes)) + 1
    final_grid    = labeled == largest_label
    n_post        = int(final_grid.sum())

    logger.info(
        "Voxelization: %d raw → %d simplified (%.1f%% reduction | %d extra component%s dropped)",
        n_pre, n_post,
        100.0 * (n_pre - n_post) / max(n_pre, 1),
        n_comp - 1, "s" if n_comp - 1 != 1 else "",
    )

    xi, yi, zi = np.where(final_grid)
    return [{"x": int(x), "y": int(y), "z": int(z)} for x, y, z in zip(xi, yi, zi)]


async def run_voxelization(run_id: str) -> dict:
    db     = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("reconstructed",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run['status']}', expected 'reconstructed'",
        )

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "voxelizing", "voxelization_started_at": datetime.now(timezone.utc)}},
    )

    try:
        recon      = run.get("reconstruction")
        stream     = await gridfs.open_download_stream(ObjectId(recon["point_cloud_file_id"]))
        point_list = json.loads(await stream.read())

        loop   = asyncio.get_event_loop()
        voxels = await loop.run_in_executor(None, _build_voxel_grid, point_list)

        if not voxels:
            raise ValueError("Voxel grid is empty after simplification — point cloud may be too sparse")

        logger.info("Voxelized run %s: %d voxels (voxel_size=%.3f)", run_id, len(voxels), VOXEL_SIZE)

        voxel_bytes   = json.dumps(voxels).encode()
        voxel_file_id = await gridfs.upload_from_stream(
            "voxels.json",
            io.BytesIO(voxel_bytes),
            metadata={"run_id": run_id, "content_type": "application/json", "stage": "voxelization"},
        )

        voxel_meta = {
            "voxel_file_id": str(voxel_file_id),
            "voxel_count":   len(voxels),
            "voxel_size":    VOXEL_SIZE,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                    "voxelized",
                "voxelization":              voxel_meta,
                "voxelization_completed_at": datetime.now(timezone.utc),
            }},
        )
        return {"run_id": run_id, "status": "voxelized", **voxel_meta}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Voxelization failed for run %s", run_id)
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
