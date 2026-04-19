"""
YOLO segmentation stage.
Reads uploaded images from GridFS, isolates the object in each frame,
saves masked outputs back to GridFS, and updates the run document.
"""
import io
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs


async def run_segmentation(run_id: str) -> dict:
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("uploaded",):
        raise HTTPException(status_code=409, detail=f"Run is in status '{run['status']}', expected 'uploaded'")

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "segmenting", "segmentation_started_at": datetime.now(timezone.utc)}},
    )

    try:
        # --- YOLO segmentation goes here ---
        # from ultralytics import YOLO
        # model = YOLO("yolov8n-seg.pt")
        # For each image in run["images"]:
        #   stream = await gridfs.open_download_stream(ObjectId(img["file_id"]))
        #   raw = await stream.read()
        #   results = model(raw)
        #   masked = results[0].plot()   # or apply mask manually
        #   store masked image back to GridFS
        # ----------------------------------

        # Placeholder: mark original images as segmented outputs until YOLO is wired in
        segmented = [
            {
                "original_file_id": img["file_id"],
                "segmented_file_id": img["file_id"],   # replace with real GridFS id after YOLO
                "filename": img["filename"],
            }
            for img in run.get("images", [])
        ]

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {
                "$set": {
                    "status": "segmented",
                    "segmented_images": segmented,
                    "segmentation_completed_at": datetime.now(timezone.utc),
                }
            },
        )
        return {"run_id": run_id, "status": "segmented", "segmented_images": segmented}

    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
