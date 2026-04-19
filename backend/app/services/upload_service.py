import io
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import UploadFile, HTTPException
from app.database import get_db, get_gridfs
from app.config import MAX_IMAGE_SIZE_MB, ALLOWED_TYPES


async def store_image(file: UploadFile, run_id: str) -> dict:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_IMAGE_SIZE_MB}MB limit")

    gridfs = get_gridfs()
    file_id = await gridfs.upload_from_stream(
        file.filename,
        io.BytesIO(data),
        metadata={"run_id": run_id, "content_type": file.content_type},
    )
    return {"file_id": str(file_id), "filename": file.filename, "size": len(data)}


async def create_run(image_files: list[UploadFile]) -> dict:
    db = get_db()
    run = {
        "status": "uploaded",
        "created_at": datetime.now(timezone.utc),
        "images": [],
    }
    result = await db.runs.insert_one(run)
    run_id = str(result.inserted_id)

    stored = []
    for f in image_files:
        meta = await store_image(f, run_id)
        stored.append(meta)

    await db.runs.update_one(
        {"_id": result.inserted_id},
        {"$set": {"images": stored}},
    )

    return {"run_id": run_id, "images": stored}


async def get_run(run_id: str) -> dict:
    db = get_db()
    try:
        obj_id = ObjectId(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")

    run = await db.runs.find_one({"_id": obj_id})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    run["run_id"] = str(run.pop("_id"))
    return run


async def stream_image(file_id: str):
    from bson import ObjectId as BsonObjectId
    gridfs = get_gridfs()
    try:
        obj_id = BsonObjectId(file_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file_id")

    try:
        stream = await gridfs.open_download_stream(obj_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

    return stream
