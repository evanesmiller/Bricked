"""
YOLO segmentation stage.

For each uploaded image:
  1. Run YOLOv8-seg to detect all objects and their masks
  2. Pick the most prominent object (largest mask area)
  3. Apply the mask — background becomes transparent (RGBA PNG)
  4. Store the masked image back to GridFS
  5. Update the run document with segmented image references
"""
import io
import asyncio
import numpy as np
from datetime import datetime, timezone
from functools import lru_cache
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs

MODEL_NAME = "yolo11x-seg.pt"  # largest YOLO11 model — best free accuracy available
CONF_THRESHOLD = 0.5          # lower than default 0.5 to catch partially visible objects


@lru_cache(maxsize=1)
def _load_model():
    """Load the YOLO model once and cache it for the lifetime of the process."""
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)


def _decode_image(image_bytes: bytes):
    """
    Decode image bytes to a BGR numpy array.
    Falls back to PIL if cv2 fails (e.g. unusual JPEG from HEIC conversion).
    """
    import cv2
    from PIL import Image

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        # PIL fallback — handles edge-case JPEG encodings cv2 rejects
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if bgr is None:
        raise ValueError("Could not decode image — file may be corrupt")

    return bgr


def _segment_image(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Run segmentation on raw image bytes.

    Returns:
        masked_png_bytes: RGBA PNG with background removed
        meta: confidence and bounding box of the selected object
    """
    import cv2
    from PIL import Image

    model = _load_model()

    bgr = _decode_image(image_bytes)
    h, w = bgr.shape[:2]

    results = model(bgr, conf=CONF_THRESHOLD, verbose=False)
    result = results[0]

    if result.masks is None or len(result.masks) == 0:
        # Retry once with an even lower threshold before giving up
        results = model(bgr, conf=0.1, verbose=False)
        result = results[0]

    if result.masks is None or len(result.masks) == 0:
        raise ValueError("No objects detected in image — ensure the object is clearly visible")

    # Pick the mask with the largest area (most prominent object)
    masks = result.masks.data.cpu().numpy()   # (N, H, W) float32 0-1
    areas = [m.sum() for m in masks]
    best_idx = int(np.argmax(areas))
    best_mask = masks[best_idx]

    # Resize mask to original image dimensions
    mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

    # Build RGBA image: object pixels keep colour, background becomes transparent
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba = np.dstack([rgb, binary_mask])
    pil_img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    masked_bytes = buf.getvalue()

    # Confidence and box for the chosen detection
    conf = float(result.boxes.conf[best_idx].cpu()) if result.boxes is not None else 0.0
    box = result.boxes.xyxy[best_idx].cpu().tolist() if result.boxes is not None else []

    return masked_bytes, {"confidence": round(conf, 4), "box": box}


async def run_segmentation(run_id: str) -> dict:
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("uploaded",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run['status']}', expected 'uploaded'",
        )

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {"status": "segmenting", "segmentation_started_at": datetime.now(timezone.utc)}},
    )

    try:
        segmented = []

        for img in run.get("images", []):
            # Load original image from GridFS
            stream = await gridfs.open_download_stream(ObjectId(img["file_id"]))
            image_bytes = await stream.read()

            # Run YOLO in a thread so we don't block the async event loop
            masked_bytes, detection_meta = await asyncio.get_event_loop().run_in_executor(
                None, _segment_image, image_bytes
            )

            # Save masked PNG back to GridFS
            seg_filename = img["filename"].rsplit(".", 1)[0] + "_seg.png"
            seg_file_id = await gridfs.upload_from_stream(
                seg_filename,
                io.BytesIO(masked_bytes),
                metadata={"run_id": run_id, "content_type": "image/png", "stage": "segmented"},
            )

            segmented.append({
                "original_file_id": img["file_id"],
                "segmented_file_id": str(seg_file_id),
                "filename": seg_filename,
                "detection": detection_meta,
            })

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

    except HTTPException:
        raise
    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
