import io
import os
from datetime import datetime, timezone
from bson import ObjectId
from fastapi import UploadFile, HTTPException
from app.database import get_db, get_gridfs
from app.config import MAX_IMAGE_SIZE_MB, ALLOWED_TYPES, HEIC_EXTENSIONS, HEIC_MIME_TYPES


def _is_heic(file: UploadFile) -> bool:
    """
    Browsers frequently send HEIC files as application/octet-stream.
    Fall back to checking the file extension.
    """
    ext = os.path.splitext(file.filename or "")[1].lower()
    return file.content_type in HEIC_MIME_TYPES or ext in HEIC_EXTENSIONS


def _convert_heic_to_jpeg(data: bytes) -> bytes:
    """Convert HEIC bytes to JPEG bytes using pillow-heif."""
    import pillow_heif
    from PIL import Image

    pillow_heif.register_heif_opener()
    img = Image.open(io.BytesIO(data))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def _resolve_content_type(file: UploadFile) -> str:
    """Return the canonical MIME type, treating octet-stream HEIC files correctly."""
    if _is_heic(file):
        return "image/heic"
    return file.content_type


async def store_image(file: UploadFile, run_id: str) -> dict:
    resolved_type = _resolve_content_type(file)

    if resolved_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}")

    data = await file.read()
    if len(data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File exceeds {MAX_IMAGE_SIZE_MB}MB limit")

    # Convert HEIC to JPEG so downstream pipeline (YOLO, OpenCV) can read it
    stored_type = resolved_type
    stored_filename = file.filename
    if resolved_type in HEIC_MIME_TYPES or _is_heic(file):
        data = _convert_heic_to_jpeg(data)
        stored_type = "image/jpeg"
        stored_filename = os.path.splitext(file.filename)[0] + ".jpg"

    gridfs = get_gridfs()
    file_id = await gridfs.upload_from_stream(
        stored_filename,
        io.BytesIO(data),
        metadata={"run_id": run_id, "content_type": stored_type},
    )
    return {
        "file_id": str(file_id),
        "filename": stored_filename,
        "original_filename": file.filename,
        "size": len(data),
        "content_type": stored_type,
    }


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
    gridfs = get_gridfs()
    try:
        obj_id = ObjectId(file_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid file_id")

    try:
        stream = await gridfs.open_download_stream(obj_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Image not found")

    return stream
