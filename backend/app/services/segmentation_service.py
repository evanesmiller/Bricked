"""
YOLO segmentation stage.

For each uploaded image:
  1. Pre-process (CLAHE contrast + unsharp sharpening) to maximise YOLO confidence
  2. Run YOLOv11-seg to detect all objects and their masks
  3. Pick the most prominent object (largest mask area)
  4. Discard images whose best detection falls below MIN_SEG_CONFIDENCE
  5. Apply the mask — background becomes transparent (RGBA PNG)
  6. Store the masked image back to GridFS
  7. Update the run document with segmented image references
"""
import io
import asyncio
import logging
import numpy as np
from datetime import datetime, timezone
from functools import lru_cache
from bson import ObjectId
from fastapi import HTTPException
from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

MODEL_NAME       = "yolo11x-seg.pt"
CONF_THRESHOLD   = 0.35   # initial detection threshold
CONF_RETRY       = 0.10   # fallback threshold if nothing found at CONF_THRESHOLD
MIN_SEG_CONFIDENCE = 0.00 # discard any detection below this after both passes


@lru_cache(maxsize=1)
def _load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)


def _decode_image(image_bytes: bytes) -> np.ndarray:
    import cv2
    from PIL import Image

    nparr = np.frombuffer(image_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if bgr is None:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if bgr is None:
        raise ValueError("Could not decode image — file may be corrupt")

    return bgr


def _preprocess_for_detection(bgr: np.ndarray) -> np.ndarray:
    """
    Enhance contrast and sharpness so YOLO receives a cleaner signal.

    Steps:
      1. CLAHE on the L channel (LAB) — lifts local contrast without
         blowing out highlights, which directly raises YOLO confidence on
         objects that blend into similar-toned backgrounds.
      2. Unsharp mask — amplifies edges so the model sees crisper boundaries,
         improving mask precision as well as detection confidence.
    """
    import cv2

    # 1 — CLAHE contrast enhancement (luminance only, preserves colour)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l), a, b])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 2 — Unsharp mask (gentle: 1.4× original − 0.4× blurred)
    blurred   = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.0)
    sharpened = cv2.addWeighted(enhanced, 1.4, blurred, -0.4, 0)

    return sharpened


def _segment_image(image_bytes: bytes) -> tuple[bytes, dict]:
    """
    Pre-process then run YOLO segmentation on raw image bytes.

    Returns:
        masked_png_bytes : RGBA PNG with background removed
        meta             : confidence, bounding box, and preprocessing flag
    Raises:
        ValueError if no object is detected or confidence is too low.
    """
    import cv2
    from PIL import Image

    model = _load_model()
    bgr   = _decode_image(image_bytes)
    h, w  = bgr.shape[:2]

    # Pre-process to boost detection confidence
    bgr_proc = _preprocess_for_detection(bgr)

    # First pass
    result = model(bgr_proc, conf=CONF_THRESHOLD, verbose=False)[0]

    # Retry with lower threshold if nothing found
    if result.masks is None or len(result.masks) == 0:
        result = model(bgr_proc, conf=CONF_RETRY, verbose=False)[0]

    if result.masks is None or len(result.masks) == 0:
        raise ValueError(
            "No objects detected — ensure the object is clearly visible, "
            "well-lit, and contrasts with the background"
        )

    # Pick mask with the largest area (most prominent object)
    masks    = result.masks.data.cpu().numpy()   # (N, H, W) float32
    areas    = [m.sum() for m in masks]
    best_idx = int(np.argmax(areas))

    conf = float(result.boxes.conf[best_idx].cpu()) if result.boxes is not None else 0.0

    if conf < MIN_SEG_CONFIDENCE:
        raise ValueError(
            f"Detection confidence too low ({conf:.2f} < {MIN_SEG_CONFIDENCE}) — "
            "retake this image with better lighting or a plainer background"
        )

    # Apply mask to the *original* (non-preprocessed) image so colour is natural
    best_mask    = masks[best_idx]
    mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    binary_mask  = (mask_resized > 0.5).astype(np.uint8) * 255

    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgba   = np.dstack([rgb, binary_mask])
    pil_img = Image.fromarray(rgba, mode="RGBA")

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")

    box = result.boxes.xyxy[best_idx].cpu().tolist() if result.boxes is not None else []
    return buf.getvalue(), {"confidence": round(conf, 4), "box": box}


async def run_segmentation(run_id: str) -> dict:
    db      = get_db()
    gridfs  = get_gridfs()

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
        segmented: list[dict] = []
        skipped:   list[dict] = []

        for img in run.get("images", []):
            stream      = await gridfs.open_download_stream(ObjectId(img["file_id"]))
            image_bytes = await stream.read()

            try:
                masked_bytes, detection_meta = await asyncio.get_event_loop().run_in_executor(
                    None, _segment_image, image_bytes
                )
            except ValueError as skip_exc:
                reason = str(skip_exc)
                logger.warning("Skipping %s: %s", img["filename"], reason)
                skipped.append({"filename": img["filename"], "reason": reason})
                continue

            seg_filename = img["filename"].rsplit(".", 1)[0] + "_seg.png"
            seg_file_id  = await gridfs.upload_from_stream(
                seg_filename,
                io.BytesIO(masked_bytes),
                metadata={"run_id": run_id, "content_type": "image/png", "stage": "segmented"},
            )

            logger.info(
                "Segmented %s  conf=%.2f",
                img["filename"], detection_meta["confidence"]
            )
            segmented.append({
                "original_file_id": img["file_id"],
                "segmented_file_id": str(seg_file_id),
                "filename": seg_filename,
                "detection": detection_meta,
            })

        if len(segmented) < 2:
            raise ValueError(
                f"Only {len(segmented)} image(s) passed the confidence threshold "
                f"(need at least 2). Skipped images: "
                + ", ".join(f"{s['filename']} ({s['reason']})" for s in skipped)
            )

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                      "segmented",
                "segmented_images":            segmented,
                "skipped_images":              skipped,
                "segmentation_completed_at":   datetime.now(timezone.utc),
            }},
        )
        return {
            "run_id":          run_id,
            "status":          "segmented",
            "segmented_images": segmented,
            "skipped_images":  skipped,
        }

    except HTTPException:
        raise
    except Exception as exc:
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
