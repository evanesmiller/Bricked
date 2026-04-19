"""
Voxelization stage using Open3D.
Reads the reconstructed point cloud, converts it into a voxel grid,
and stores the voxel data in MongoDB + GridFS.
"""
import asyncio
import io
import json
import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import HTTPException

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

VOXEL_SIZE = 0.05  # world units in [-1, 1]³ → ~40 voxels per axis max


def _build_voxel_grid(point_list: list[dict]) -> list[dict]:
    import numpy as np
    import open3d as o3d

    if not point_list:
        return []

    pts = np.array([[p["x"], p["y"], p["z"]] for p in point_list], dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)

    vg = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=VOXEL_SIZE)
    return [
        {"x": int(v.grid_index[0]), "y": int(v.grid_index[1]), "z": int(v.grid_index[2])}
        for v in vg.get_voxels()
    ]


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
        recon  = run.get("reconstruction")
        stream = await gridfs.open_download_stream(ObjectId(recon["point_cloud_file_id"]))
        point_list = json.loads(await stream.read())

        loop   = asyncio.get_event_loop()
        voxels = await loop.run_in_executor(None, _build_voxel_grid, point_list)

        if not voxels:
            raise ValueError("Voxel grid is empty — point cloud may be too sparse")

        logger.info(
            "Voxelized run %s: %d voxels (voxel_size=%.3f)",
            run_id, len(voxels), VOXEL_SIZE,
        )

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
                "status":                     "voxelized",
                "voxelization":               voxel_meta,
                "voxelization_completed_at":  datetime.now(timezone.utc),
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
