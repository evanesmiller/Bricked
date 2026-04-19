from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from app.services import upload_service

router = APIRouter(prefix="/uploads", tags=["uploads"])


@router.post("/runs", status_code=201)
async def create_run(images: list[UploadFile] = File(...)):
    """Upload 1–20 images and create a new processing run."""
    if not images:
        raise HTTPException(status_code=400, detail="At least one image is required")
    if len(images) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 images per run")
    return await upload_service.create_run(images)


@router.get("/runs/{run_id}")
async def get_run(run_id: str):
    """Return run metadata and image references."""
    return await upload_service.get_run(run_id)


@router.get("/images/{file_id}")
async def get_image(file_id: str):
    """Stream a stored image by its GridFS file_id."""
    stream = await upload_service.stream_image(file_id)
    content_type = stream.metadata.get("content_type", "image/jpeg") if stream.metadata else "image/jpeg"

    async def iter_chunks():
        while True:
            chunk = await stream.readchunk()
            if not chunk:
                break
            yield chunk

    return StreamingResponse(iter_chunks(), media_type=content_type)
