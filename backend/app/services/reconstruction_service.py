"""
COLMAP-based reconstruction with visual hull carving.

Pipeline:
  1. Download original (unsegmented) images to a temp workspace
  2. Run COLMAP: feature extraction → exhaustive matching → incremental SfM
  3. Extract real camera matrices (K, R, t) and sparse point cloud
  4. Download segmented RGBA images and extract binary silhouettes
  5. Carve a voxel grid using ALL views:
       - COLMAP-registered: real camera pose + intrinsics
       - Unregistered: turntable assumption in COLMAP world space
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
from bson import ObjectId
from fastapi import HTTPException

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

GRID_SIZE        = 96    # voxel grid resolution (N³)
DILATION_PX      = 6     # silhouette dilation for COLMAP views
MIN_VOTE_FRAC    = 0.70  # fraction of views a voxel must pass to survive

# Turntable fallback parameters (for unregistered images)
CAMERA_DISTANCE  = 3.0
CAMERA_ELEVATION = 0.3
NORM_SIZE        = 256
NORM_PAD         = 0.10
DILATION_PX_SIL  = 10

# iPhone photos are 4284×4284 — resize before COLMAP to avoid 15-min extraction
# and numerically unstable structure-less resection at large pixel coordinates.
COLMAP_MAX_DIM   = 1600


def _resize_for_colmap(image_bytes: bytes, max_dim: int = COLMAP_MAX_DIM) -> bytes:
    """Downscale image so its longest side ≤ max_dim; return JPEG bytes."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return image_bytes
    scale  = max_dim / max(h, w)
    img    = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else image_bytes


# ── Camera intrinsics helper ──────────────────────────────────────────────────

def _get_intrinsics(camera) -> tuple[float, float, float, float]:
    """Return (fx, fy, cx, cy) from a pycolmap Camera regardless of model."""
    params = camera.params
    name = str(camera.model).upper()
    if "SIMPLE" in name:
        return float(params[0]), float(params[0]), float(params[1]), float(params[2])
    else:
        return float(params[0]), float(params[1]), float(params[2]), float(params[3])


# ── Silhouette helpers ────────────────────────────────────────────────────────

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


def _normalize_silhouette(mask: np.ndarray) -> np.ndarray:
    """Crop to bounding box, pad, resize to NORM_SIZE, dilate."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.zeros((NORM_SIZE, NORM_SIZE), dtype=np.uint8)

    h_img, w_img = mask.shape
    r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])

    pad_r = max(1, int((r1 - r0) * NORM_PAD))
    pad_c = max(1, int((c1 - c0) * NORM_PAD))
    r0 = max(0, r0 - pad_r);  r1 = min(h_img, r1 + pad_r)
    c0 = max(0, c0 - pad_c);  c1 = min(w_img, c1 + pad_c)

    side = max(r1 - r0, c1 - c0)
    cr   = (r0 + r1) // 2;  cc = (c0 + c1) // 2
    r0   = max(0, cr - side // 2);  r1 = min(h_img, r0 + side)
    c0   = max(0, cc - side // 2);  c1 = min(w_img, c0 + side)

    cropped = mask[r0:r1, c0:c1]
    if 0 in cropped.shape:
        return np.zeros((NORM_SIZE, NORM_SIZE), dtype=np.uint8)

    normed = cv2.resize(cropped, (NORM_SIZE, NORM_SIZE), interpolation=cv2.INTER_NEAREST)

    if DILATION_PX_SIL > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (DILATION_PX_SIL * 2 + 1,) * 2
        )
        normed = cv2.dilate(normed, kernel)

    return normed


# ── Turntable camera builder ──────────────────────────────────────────────────

def _build_camera(angle: float, elevation: float, distance: float):
    """Return (R, t) for a turntable camera at the given position."""
    cam_pos = np.array([
        distance * np.cos(elevation) * np.sin(angle),
        distance * np.sin(elevation),
        distance * np.cos(elevation) * np.cos(angle),
    ])
    z_axis  = -cam_pos / np.linalg.norm(cam_pos)
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_axis, world_up)) > 0.99:
        world_up = np.array([0.0, 0.0, 1.0])
    x_axis  = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis  = np.cross(z_axis, x_axis)
    R = np.stack([x_axis, y_axis, z_axis], axis=0)
    t = -R @ cam_pos
    return R, t


# ── World-up estimation for turntable alignment ───────────────────────────────

def _estimate_world_up(reconstruction) -> np.ndarray:
    """Average -R[1,:] across registered cameras to estimate real-world up axis."""
    up_vectors = []
    for colmap_img in reconstruction.images.values():
        pose = colmap_img.cam_from_world()
        R = pose.rotation.matrix()
        up_vectors.append(-R[1, :])  # COLMAP Y points down in camera space
    up = np.mean(up_vectors, axis=0)
    norm = np.linalg.norm(up)
    return (up / norm).astype(np.float32) if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _align_rotation(world_up: np.ndarray) -> np.ndarray:
    """Rodrigues rotation mapping canonical [0,1,0] → world_up."""
    a = np.array([0.0, 1.0, 0.0])
    b = world_up / np.linalg.norm(world_up)
    cross = np.cross(a, b)
    dot = float(np.dot(a, b))
    s = np.linalg.norm(cross)
    if s < 1e-6:
        return np.eye(3, dtype=np.float32) if dot > 0 else np.diag([-1.0, 1.0, -1.0]).astype(np.float32)
    K = np.array([[0, -cross[2], cross[1]],
                  [cross[2], 0, -cross[0]],
                  [-cross[1], cross[0], 0]], dtype=np.float64)
    R = np.eye(3) + K + K @ K * ((1.0 - dot) / (s * s))
    return R.astype(np.float32)


# ── COLMAP runner ─────────────────────────────────────────────────────────────

def _run_colmap(images_dir: Path, workspace: Path):
    """
    Run COLMAP feature extraction, exhaustive matching, and incremental mapping.
    Returns the best Reconstruction (most registered images).
    Raises ValueError if reconstruction fails or fewer than 2 images register.
    """
    import pycolmap

    db_path    = workspace / "colmap.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    logger.info("COLMAP: extracting features from %s", images_dir)
    extraction_options = pycolmap.FeatureExtractionOptions()
    extraction_options.sift.max_num_features = 16384   # default 8192 — helps low-texture surfaces
    pycolmap.extract_features(
        database_path=str(db_path),
        image_path=str(images_dir),
        camera_mode=pycolmap.CameraMode.PER_IMAGE,
        extraction_options=extraction_options,
    )

    logger.info("COLMAP: exhaustive feature matching")
    pycolmap.match_exhaustive(database_path=str(db_path))

    logger.info("COLMAP: incremental mapping")
    mapper_options = pycolmap.IncrementalPipelineOptions()
    mapper_options.min_num_matches                 = 10  # default 15; raised from 5 to avoid false registrations
    mapper_options.mapper.init_min_num_inliers     = 25  # default 100
    mapper_options.mapper.abs_pose_min_num_inliers = 15  # raised from 10; prevents structure-less fallback SIGSEGV
    maps = pycolmap.incremental_mapping(
        database_path=str(db_path),
        image_path=str(images_dir),
        output_path=str(sparse_dir),
        options=mapper_options,
    )

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


# ── Unified visual hull carving ───────────────────────────────────────────────

def _visual_hull_unified(
    camera_views: list[tuple],   # (R, t, fx, fy, cx, cy, w, h, mask) per view
    pts_world: np.ndarray,       # (N, 3) voxel centres in world coords
) -> np.ndarray:
    """
    Carve pts_world using every camera view in camera_views.
    Returns boolean occupied mask over pts_world.
    """
    n_views = len(camera_views)
    votes   = np.zeros(len(pts_world), dtype=np.int32)

    for R, t, fx, fy, cx, cy, w, h, mask in camera_views:
        pts_cam  = (R @ pts_world.T).T + t
        z        = pts_cam[:, 2]
        in_front = z > 1e-4

        safe_z = np.where(in_front, z, 1.0)
        px = fx * pts_cam[:, 0] / safe_z + cx
        py = fy * pts_cam[:, 1] / safe_z + cy

        px_i = np.round(px).astype(np.int32)
        py_i = np.round(py).astype(np.int32)

        in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)
        in_sil    = np.zeros(len(pts_world), dtype=bool)
        idx = np.where(in_bounds)[0]
        if idx.size:
            in_sil[idx] = mask[py_i[idx], px_i[idx]] > 0

        votes += (in_sil | ~in_front).astype(np.int32)

    min_votes = max(1, int(np.ceil(n_views * MIN_VOTE_FRAC)))
    return pts_world[votes >= min_votes]


# ── Async pipeline stage ──────────────────────────────────────────────────────

def _build_camera_views(
    reconstruction,
    silhouettes: dict[str, np.ndarray],
    orig_images: list[dict],
    colmap_to_orig: dict[str, str] | None = None,
) -> tuple[list[tuple], np.ndarray, np.ndarray]:
    """
    Build the list of camera views for unified carving.

    COLMAP-registered images use real camera matrices and intrinsics.
    Unregistered images use turntable cameras placed in COLMAP world space.

    Returns (camera_views, pts_world, center) where pts_world is the voxel grid
    in COLMAP world space and center is the sparse-cloud centroid.
    """
    sparse = np.array(
        [pt.xyz for pt in reconstruction.points3D.values()], dtype=np.float32
    )
    center = sparse.mean(axis=0)
    extent = (sparse.max(axis=0) - sparse.min(axis=0)).max()
    scale  = extent / 2.0 * 1.35   # 35 % margin

    world_min = center - scale
    world_max = center + scale
    axes = [np.linspace(world_min[i], world_max[i], GRID_SIZE, dtype=np.float32)
            for i in range(3)]
    gx, gy, gz = np.meshgrid(axes[0], axes[1], axes[2], indexing="ij")
    pts_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    world_up = _estimate_world_up(reconstruction)
    align_R  = _align_rotation(world_up)

    camera_views: list[tuple] = []
    registered_names: set[str] = set()

    # ── COLMAP-registered views ───────────────────────────────────────────────
    for colmap_img in reconstruction.images.values():
        colmap_fname = colmap_img.name
        # Resolve back to original filename (pre-resize) for silhouette lookup
        orig_fname = (colmap_to_orig or {}).get(colmap_fname, colmap_fname)
        registered_names.add(orig_fname)
        if orig_fname not in silhouettes:
            logger.warning("No silhouette for COLMAP image '%s' — skipping", colmap_fname)
            continue

        mask  = silhouettes[orig_fname].copy()
        cam   = reconstruction.cameras[colmap_img.camera_id]
        fx, fy, cx, cy = _get_intrinsics(cam)
        cam_w, cam_h   = cam.width, cam.height

        if mask.shape[0] != cam_h or mask.shape[1] != cam_w:
            mask = cv2.resize(mask, (cam_w, cam_h), interpolation=cv2.INTER_NEAREST)

        pose = colmap_img.cam_from_world()
        R = pose.rotation.matrix().astype(np.float32)
        t = pose.translation.astype(np.float32)

        camera_views.append((R, t, fx, fy, cx, cy, cam_w, cam_h, mask))

    # ── Turntable views for unregistered images ───────────────────────────────
    n_total = len(orig_images)
    # Focal length: object spans ±scale at distance CAMERA_DISTANCE*scale
    # Same formula as pure-silhouette path (scale cancels out)
    f_sil = (0.5 - NORM_PAD) * NORM_SIZE * CAMERA_DISTANCE
    cx_sil = cy_sil = NORM_SIZE / 2.0

    n_fallback = 0
    for i, img_meta in enumerate(orig_images):
        fname = img_meta["filename"]
        if fname in registered_names or fname not in silhouettes:
            continue

        norm_mask = _normalize_silhouette(silhouettes[fname])
        azimuth   = 2.0 * np.pi * i / n_total

        R_turn, t_unit = _build_camera(azimuth, CAMERA_ELEVATION, CAMERA_DISTANCE)
        # Rotate turntable cameras to match COLMAP world up axis
        R_eff    = (R_turn.astype(np.float64) @ align_R.T.astype(np.float64)).astype(np.float32)
        t_colmap = (scale * t_unit - R_eff @ center).astype(np.float32)

        camera_views.append((R_eff, t_colmap, f_sil, f_sil, cx_sil, cy_sil,
                              NORM_SIZE, NORM_SIZE, norm_mask))
        n_fallback += 1

    logger.info(
        "Camera views: %d COLMAP-registered, %d turntable fallback",
        len(camera_views) - n_fallback, n_fallback,
    )
    return camera_views, pts_world, center


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
            raise ValueError("Need at least 2 original images for reconstruction")
        if len(seg_images) < 2:
            raise ValueError("Need at least 2 segmented images for visual hull carving")

        loop = asyncio.get_event_loop()

        file_id_to_name = {img["file_id"]: img["filename"] for img in orig_images}
        # Map the normalised COLMAP filename (stem + .jpg) → original filename
        colmap_to_orig  = {
            img["filename"].rsplit(".", 1)[0] + ".jpg": img["filename"]
            for img in orig_images
        }

        tmpdir = Path(tempfile.mkdtemp(prefix="bricked_colmap_"))
        try:
            images_dir = tmpdir / "images"
            images_dir.mkdir()

            for img_meta in orig_images:
                stream = await fs.open_download_stream(ObjectId(img_meta["file_id"]))
                data   = await stream.read()
                data   = await loop.run_in_executor(None, _resize_for_colmap, data)
                fname  = img_meta["filename"].rsplit(".", 1)[0] + ".jpg"
                (images_dir / fname).write_bytes(data)

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
                    "Could not match any segmented images to original filenames."
                )

            logger.info("Starting COLMAP for run %s (%d images)", run_id, len(orig_images))
            reconstruction = await loop.run_in_executor(
                None, _run_colmap, images_dir, tmpdir
            )

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        n_registered = len(reconstruction.images)

        # Build unified camera views (COLMAP + turntable fallback)
        camera_views, pts_world, _ = await loop.run_in_executor(
            None, _build_camera_views, reconstruction, silhouettes, orig_images, colmap_to_orig
        )

        if not camera_views:
            raise ValueError("No camera views available for carving.")

        logger.info(
            "Starting unified visual hull carving for run %s (%d views total)",
            run_id, len(camera_views),
        )
        pts = await loop.run_in_executor(
            None, _visual_hull_unified, camera_views, pts_world
        )

        if len(pts) == 0:
            raise ValueError(
                "Visual hull is empty — silhouettes may not overlap. "
                "Try more images from additional angles."
            )

        logger.info("Visual hull: %d occupied voxels for run %s", len(pts), run_id)

        # Normalise to [-1, 1]³
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

        meta = {
            "point_cloud_file_id":      str(cloud_id),
            "point_count":              len(point_list),
            "method":                   "colmap_visual_hull",
            "colmap_registered_images": n_registered,
            "colmap_sparse_points":     len(reconstruction.points3D),
            "total_carving_views":      len(camera_views),
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
