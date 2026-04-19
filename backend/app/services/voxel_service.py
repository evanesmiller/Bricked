"""
Voxelization stage using Open3D.
Reads the reconstructed point cloud, converts it into a voxel grid,
and stores the voxel data in MongoDB + GridFS.
"""
import json
import io
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs


async def run_voxelization(run_id: str) -> dict:
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("reconstructed",):
        raise HTTPException(status_code=409, detail=f"Run is in status '{run['status']}', expected 'reconstructed'")

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "voxelizing", "voxelization_started_at": datetime.now(timezone.utc)}},
    )

    try:
        # --- Open3D voxelization goes here ---
        # import open3d as o3d, numpy as np
        #
        # stream = await gridfs.open_download_stream(ObjectId(run["reconstruction"]["point_cloud_file_id"]))
        # pts = json.loads(await stream.read())
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(np.array([[p["x"], p["y"], p["z"]] for p in pts]))
        #
        # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.05)
        # voxels = [{"x": v.grid_index[0], "y": v.grid_index[1], "z": v.grid_index[2]}
        #           for v in voxel_grid.get_voxels()]
        # ------------------------------------

        # Placeholder: 3x3x3 solid voxel grid
        voxels = [
            {"x": x, "y": y, "z": z}
            for x in range(3) for y in range(3) for z in range(3)
        ]
        voxel_bytes = json.dumps(voxels).encode()
        voxel_file_id = await gridfs.upload_from_stream(
            "voxels.json",
            io.BytesIO(voxel_bytes),
            metadata={"run_id": run_id, "content_type": "application/json"},
        )

        voxel_meta = {
            "voxel_file_id": str(voxel_file_id),
            "voxel_count": len(voxels),
            "voxel_size": 0.05,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {
                "$set": {
                    "status": "voxelized",
                    "voxelization": voxel_meta,
                    "voxelization_completed_at": datetime.now(timezone.utc),
                }
            },
        )
        return {"run_id": run_id, "status": "voxelized", **voxel_meta}

    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
