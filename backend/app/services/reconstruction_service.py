"""
Structure-from-Motion (SfM) reconstruction stage using OpenCV.
Reads segmented images, extracts keypoints, matches features across views,
estimates a coarse 3D point cloud, and stores metadata in MongoDB.
"""
import json
import io
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs


async def run_reconstruction(run_id: str) -> dict:
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("segmented",):
        raise HTTPException(status_code=409, detail=f"Run is in status '{run['status']}', expected 'segmented'")

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "reconstructing", "reconstruction_started_at": datetime.now(timezone.utc)}},
    )

    try:
        # --- OpenCV SfM goes here ---
        # import cv2, numpy as np
        # images = []
        # for img_meta in run["segmented_images"]:
        #     stream = await gridfs.open_download_stream(ObjectId(img_meta["segmented_file_id"]))
        #     buf = np.frombuffer(await stream.read(), dtype=np.uint8)
        #     images.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
        #
        # sift = cv2.SIFT_create()
        # keypoints, descriptors, point_cloud = sfm_pipeline(images, sift)
        #
        # Save point cloud JSON to GridFS
        # pts_bytes = json.dumps(point_cloud.tolist()).encode()
        # cloud_id = await gridfs.upload_from_stream("point_cloud.json", io.BytesIO(pts_bytes), ...)
        # ----------------------------

        # Placeholder point cloud (cube of 8 points)
        placeholder_cloud = [
            {"x": x, "y": y, "z": z}
            for x in (0, 1) for y in (0, 1) for z in (0, 1)
        ]
        pts_bytes = json.dumps(placeholder_cloud).encode()
        cloud_id = await gridfs.upload_from_stream(
            "point_cloud.json",
            io.BytesIO(pts_bytes),
            metadata={"run_id": run_id, "content_type": "application/json"},
        )

        reconstruction_meta = {
            "point_cloud_file_id": str(cloud_id),
            "point_count": len(placeholder_cloud),
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {
                "$set": {
                    "status": "reconstructed",
                    "reconstruction": reconstruction_meta,
                    "reconstruction_completed_at": datetime.now(timezone.utc),
                }
            },
        )
        return {"run_id": run_id, "status": "reconstructed", **reconstruction_meta}

    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
