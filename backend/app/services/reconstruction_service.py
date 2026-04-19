"""
COLMAP-based reconstruction with visual hull carving.

Pipeline:
  1. Download original (unsegmented) images to a temp workspace
  2. Run COLMAP: feature extraction → exhaustive matching → incremental SfM
  3. Extract real camera matrices (K, R, t) and sparse point cloud
  4. Download segmented RGBA images and extract binary silhouettes
  5. Carve a voxel grid using real camera projections + silhouette masks
  6. Normalise resulting point cloud to [-1, 1]³ and store in GridFS
"""

import asyncio
import io
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import pycolmap
from bson import ObjectId
from fastapi import HTTPException

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

GRID_SIZE      = 64    # voxel grid resolution (N³)
DILATION_PX    = 6     # silhouette dilation for projection-error tolerance
MIN_VOTE_FRAC  = 0.70  # fraction of views a voxel must pass to survive


# ── Camera intrinsics helper ──────────────────────────────────────────────────

def _get_intrinsics(camera) -> tuple[float, float, float, float]:
    """
    Return (fx, fy, cx, cy) from a pycolmap Camera regardless of model.
    Handles SIMPLE_PINHOLE, SIMPLE_RADIAL, PINHOLE, RADIAL, and variants.
    """
    params = camera.params
    name = str(camera.model).upper()
    if "SIMPLE" in name:
        # params = [f, cx, cy, ...]
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    else:
        # PINHOLE / RADIAL: params = [fx, fy, cx, cy, ...]
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])


# ── Silhouette extraction ─────────────────────────────────────────────────────

def _extract_silhouette(rgba_bytes: bytes, dilation_px: int = DILATION_PX) -> np.ndarray:
    """Decode RGBA PNG → binary mask (H×W uint8 0/1)."""
    buf = np.frombuffer(rgba_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode segmented image")

    if img.ndim == 3 and img.shape[2] == 4:
        mask = (img[:, :, 3] > 32).astype(np.uint8)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, mask = cv2.threshold(gray, 250, 1, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if dilation_px > 0:
        dk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1,) * 2)
        mask = cv2.dilate(mask, dk)

    return mask


# ── COLMAP runner ─────────────────────────────────────────────────────────────

def _run_colmap(images_dir: Path, workspace: Path) -> pycolmap.Reconstruction:
    """
    Run COLMAP feature extraction, exhaustive matching, and incremental mapping.
    Returns the best Reconstruction (most registered images).
    Raises ValueError if reconstruction fails or fewer than 2 images register.
    """
    db_path   = workspace / "colmap.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    logger.info("COLMAP: extracting features from %s", images_dir)
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,  # all images share one camera model
    )

    logger.info("COLMAP: exhaustive feature matching")
    pycolmap.match_exhaustive(database_path=str(db_path))

    logger.info("COLMAP: incremental mapping")
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
    )

    # maps may be a list or a dict depending on pycolmap version
    recon_list = list(maps.values()) if isinstance(maps, dict) else list(maps)
    if not recon_list:
        raise ValueError(
            "COLMAP failed to reconstruct any camera positions. "
            "Ensure the object has visible surface texture and all photos overlap clearly."
        )

    best = max(recon_list, key=lambda r: len(r.images))
    n_registered = len(best.images)

    if n_registered < 2:
        raise ValueError(
            f"COLMAP only registered {n_registered} image(s) — need at least 2. "
            "Try a more textured object, better lighting, or more image overlap."
        )

    logger.info(
        "COLMAP: registered %d images, %d sparse 3-D points",
        n_registered, len(best.points3D),
    )
    return best


# ── Visual hull with real camera matrices ─────────────────────────────────────

def _visual_hull_colmap(
    reconstruction: pycolmap.Reconstruction,
    silhouettes: dict[str, np.ndarray],   # original_filename → binary mask
    grid_size: int = GRID_SIZE,
) -> np.ndarray:
    """
    Carve a voxel grid bounded by the COLMAP sparse point cloud using real
    camera matrices recovered by COLMAP.

    Returns an (M, 3) float32 array of occupied voxel centres in COLMAP
    world coordinates.
    """
    # ── Bounding box from sparse point cloud ─────────────────────────────────
    if len(reconstruction.points3D) < 3:
        raise ValueError(
            "COLMAP sparse point cloud is too sparse (< 3 points) to define a "
            "reconstruction volume. The object may not have enough texture."
        )

    sparse = np.array([pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32)
    center = sparse.mean(axis=0)
    extent = (sparse.max(axis=0) - sparse.min(axis=0)).max()
    half   = extent / 2.0 * 1.35  # 35 % margin around sparse cloud

    world_min = center - half
    world_max = center + half

    # ── Build voxel grid ──────────────────────────────────────────────────────
    axes = [np.linspace(world_min[i], world_max[i], grid_size, dtype=np.float32)
            for i in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pts_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)  # (N,3)

    votes   = np.zeros(grid_size ** 3, dtype=np.int32)
    n_views = 0

    # ── Project each registered image ─────────────────────────────────────────
    for colmap_img in reconstruction.images.values():
        fname = colmap_img.name
        if fname not in silhouettes:
            logger.warning("No silhouette for COLMAP image '%s' — skipping", fname)
            continue

        mask = silhouettes[fname]          # (H, W) uint8

        camera = reconstruction.cameras[colmap_img.camera_id]
        fx, fy, cx, cy = _get_intrinsics(camera)
        cam_w, cam_h   = camera.width, camera.height

        # Resize silhouette to match COLMAP camera resolution if needed
        sil_h, sil_w = mask.shape
        if sil_h != cam_h or sil_w != cam_w:
            mask = cv2.resize(mask, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)

        R = colmap_img.cam_from_world.rotation.matrix().astype(np.float32)
        t = colmap_img.cam_from_world.translation.astype(np.float32)

        # World → camera space
        pts_cam = (R @ pts_world.T).T + t   # (N, 3)
        z       = pts_cam[:, 2]
        in_front = z > 1e-4

        safe_z = np.where(in_front, z, 1.0)
        px = fx * pts_cam[:, 0] / safe_z + cx
        py = fy * pts_cam[:, 1] / safe_z + cy

        px_i = np.round(px).astype(np.int32)
        py_i = np.round(py).astype(np.int32)

        in_bounds = in_front & (px_i >= 0) & (px_i < cam_w) & (py_i >= 0) & (py_i < cam_h)

        in_sil = np.zeros(len(pts_world), dtype=bool)
        idx = np.where(in_bounds)[0]
        if idx.size:
            in_sil[idx] = mask[py_i[idx], px_i[idx]] > 0

        # Behind-camera voxels abstain (counted as passing this view)
        votes += (in_sil | ~in_front).astype(np.int32)
        n_views += 1

    if n_views == 0:
        raise ValueError(
            "No silhouettes matched any registered COLMAP image. "
            "Check that filenames in segmented_images align with uploaded originals."
        )

    min_votes = max(1, int(np.ceil(n_views * MIN_VOTE_FRAC)))
    occupied  = pts_world[votes >= min_votes]

    if len(occupied) == 0:
        raise ValueError(
            "Visual hull is empty after carving. "
            "The silhouettes may not be consistent with the COLMAP camera poses."
        )

    logger.info("Visual hull: %d occupied voxels from %d views", len(occupied), n_views)
    return occupied


# ── Async pipeline stage ──────────────────────────────────────────────────────

async def run_reconstruction(run_id: str) -> dict:
    db = get_db()
    fs = get_gridfs()

    run = await db.runs.find_one({"_id": ObjectId(run_id)})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run["status"] not in ("segmented",):
        raise HTTPException(
            status_code=409,
            detail=f"Run is in status '{run['status']}', expected 'segmented'",
        )

    await db.runs.update_one(
        {"_id": ObjectId(run_id)},
        {"$set": {
            "status": "reconstructing",
            "reconstruction_started_at": datetime.now(timezone.utc),
        }},
    )

    try:
        orig_images = run.get("images", [])
        seg_images  = run.get("segmented_images", [])

        if len(orig_images) < 2:
            raise ValueError("Need at least 2 original images for COLMAP reconstruction")
        if len(seg_images) < 2:
            raise ValueError("Need at least 2 segmented images for visual hull carving")

        loop = asyncio.get_event_loop()

        # Map original_file_id → original filename (used by COLMAP on disk)
        file_id_to_name = {img["file_id"]: img["filename"] for img in orig_images}

        tmpdir = Path(tempfile.mkdtemp(prefix="bricked_colmap_"))
        try:
            images_dir = tmpdir / "images"
            images_dir.mkdir()

            # ── Write original images to disk for COLMAP ──────────────────────
            for img_meta in orig_images:
                stream = await fs.open_download_stream(ObjectId(img_meta["file_id"]))
                data   = await stream.read()
                (images_dir / img_meta["filename"]).write_bytes(data)

            # ── Load silhouettes keyed by original filename ────────────────────
            silhouettes: dict[str, np.ndarray] = {}
            for seg_meta in seg_images:
                orig_fname = file_id_to_name.get(seg_meta.get("original_file_id", ""))
                if not orig_fname:
                    continue
                stream = await fs.open_download_stream(ObjectId(seg_meta["segmented_file_id"]))
                data   = await stream.read()
                mask   = await loop.run_in_executor(None, _extract_silhouette, data)
                silhouettes[orig_fname] = mask

            if not silhouettes:
                raise ValueError(
                    "Could not match any segmented images to original filenames. "
                    "The run document may be missing original_file_id on segmented_images."
                )

            # ── Run COLMAP (CPU-heavy — off the event loop) ───────────────────
            logger.info("Starting COLMAP for run %s (%d images)", run_id, len(orig_images))
            reconstruction = await loop.run_in_executor(
                None, _run_colmap, images_dir, tmpdir
            )

            # ── Visual hull carving ───────────────────────────────────────────
            logger.info("Starting visual hull carving for run %s", run_id)
            pts = await loop.run_in_executor(
                None, _visual_hull_colmap, reconstruction, silhouettes
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        # ── Normalise to [-1, 1]³ for consistent downstream + viewer ─────────
        mins   = pts.min(axis=0)
        maxs   = pts.max(axis=0)
        center = (mins + maxs) / 2.0
        scale  = (maxs - mins).max() / 2.0
        pts_norm = (pts - center) / scale if scale > 0 else pts

        point_list = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])}
                      for p in pts_norm]
        pts_bytes  = json.dumps(point_list).encode()

        cloud_id = await fs.upload_from_stream(
            "point_cloud.json",
            io.BytesIO(pts_bytes),
            metadata={
                "run_id":       run_id,
                "content_type": "application/json",
                "stage":        "reconstruction",
            },
        )

        n_registered = len(reconstruction.images)
        meta = {
            "point_cloud_file_id":      str(cloud_id),
            "point_count":              len(point_list),
            "method":                   "colmap_visual_hull",
            "colmap_registered_images": n_registered,
            "colmap_sparse_points":     len(reconstruction.points3D),
            "grid_size":                GRID_SIZE,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                      "reconstructed",
                "reconstruction":              meta,
                "reconstruction_completed_at": datetime.now(timezone.utc),
            }},
        )
        return {"run_id": run_id, "status": "reconstructed", **meta}

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Reconstruction failed for run %s", run_id)
        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {"status": "failed", "error": str(exc)}},
        )
        raise HTTPException(status_code=500, detail=str(exc))
