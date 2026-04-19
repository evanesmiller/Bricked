"""
Silhouette-based 3D reconstruction (Visual Hull / Space Carving).

For each segmented RGBA image, the object silhouette is back-projected through
an assumed turntable camera. Voxels that project outside every silhouette are
carved away; the remainder forms the visual hull, stored as a point cloud.

Camera assumption: N cameras equally spaced in azimuth on a circle around the
Y-axis at a fixed radius and slight downward elevation — a common handheld
multi-photo setup.
"""

import asyncio
import io
import json
import logging
from datetime import datetime, timezone

import cv2
import numpy as np
from bson import ObjectId
from fastapi import HTTPException
from scipy.ndimage import label as nd_label

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

# ── Carving parameters ────────────────────────────────────────────────────────

GRID_SIZE        = 256   # Voxel grid resolution (N³)
CAMERA_DISTANCE  = 3.0   # Distance from camera to object origin; object spans [-1,1]³
CAMERA_ELEVATION = 0.3   # Radians (~17°) — cameras tilt slightly downward

# Silhouette normalisation — each raw mask is cropped to its bounding box,
# padded, and resized to a square before projection.  The focal length is then
# derived analytically so the ±1 world extent maps to the padded boundary.
NORM_SIZE        = 512   # Normalised silhouette side length in pixels (2× grid for sub-voxel accuracy)
NORM_PAD         = 0.10  # Fractional padding added around the bounding box
DILATION_PX      = 3     # Extra dilation on the normalised mask (error margin)
THIN_OPEN_PX     = 4     # Opening kernel radius to disconnect thin protrusions (straws, stems)

# Chunked projection keeps peak RAM under ~400 MB at 256³ (16 M voxels × 3 × 4 B = 192 MB base)
_PROJ_CHUNK      = 4_000_000

# A voxel is kept when it lies inside at least this fraction of views.
# Higher = tighter hull, lower = more tolerant of camera model errors.
MIN_VOTE_FRAC    = 0.75


# ── Core silhouette helpers ───────────────────────────────────────────────────

def _extract_silhouette(rgba_bytes: bytes) -> tuple[np.ndarray, tuple[int, int]]:
    """Decode an RGBA PNG and return (binary_mask H×W uint8, (width, height))."""
    buf = np.frombuffer(rgba_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode segmented image")

    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        mask = (alpha > 32).astype(np.uint8)
    else:
        # Fallback: any non-white pixel is foreground
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        _, mask = cv2.threshold(gray, 250, 1, cv2.THRESH_BINARY_INV)

    # Small morphological close to fill fringe holes from YOLO mask borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    h, w = mask.shape
    return mask, (w, h)


def _build_camera(
    angle: float,
    elevation: float,
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (R, t) for a turntable camera.
    Camera sits on a circle of given radius at the given elevation angle and
    looks at the world origin.  Convention: X_cam = R @ X_world + t.
    """
    cam_pos = np.array([
        distance * np.cos(elevation) * np.sin(angle),
        distance * np.sin(elevation),
        distance * np.cos(elevation) * np.cos(angle),
    ])

    z_axis = -cam_pos / np.linalg.norm(cam_pos)          # points into scene
    world_up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(z_axis, world_up)) > 0.99:              # near-vertical view
        world_up = np.array([0.0, 0.0, 1.0])
    x_axis = np.cross(world_up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=0)        # (3, 3)
    t = -R @ cam_pos                                       # (3,)
    return R, t


# ── Visual Hull carving ───────────────────────────────────────────────────────

def _normalize_silhouette(
    mask: np.ndarray,
    out_size: int    = NORM_SIZE,
    pad_frac: float  = NORM_PAD,
    dilation_px: int = DILATION_PX,
    open_px: int     = THIN_OPEN_PX,
) -> np.ndarray:
    """
    Crop to the object bounding box, add padding, resize to a square, then
    optionally apply opening to remove thin protrusions (straws, stems), and
    finally dilate for a tolerance margin.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return np.zeros((out_size, out_size), dtype=np.uint8)

    r0, r1 = int(np.where(rows)[0][0]),  int(np.where(rows)[0][-1])
    c0, c1 = int(np.where(cols)[0][0]),  int(np.where(cols)[0][-1])
    h_img, w_img = mask.shape

    pad_r = max(1, int((r1 - r0) * pad_frac))
    pad_c = max(1, int((c1 - c0) * pad_frac))
    r0 = max(0, r0 - pad_r);  r1 = min(h_img, r1 + pad_r)
    c0 = max(0, c0 - pad_c);  c1 = min(w_img, c1 + pad_c)

    # Make square around centroid so aspect ratio doesn't distort projection
    side = max(r1 - r0, c1 - c0)
    cr   = (r0 + r1) // 2;  cc = (c0 + c1) // 2
    r0   = max(0, cr - side // 2);  r1 = min(h_img, r0 + side)
    c0   = max(0, cc - side // 2);  c1 = min(w_img, c0 + side)

    cropped = mask[r0:r1, c0:c1]
    if 0 in cropped.shape:
        return np.zeros((out_size, out_size), dtype=np.uint8)

    normed = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # Remove thin protrusions (straws, stems) via opening then keep largest component.
    # This prevents inconsistent straw positions across views from warping the hull.
    if open_px > 0:
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px * 2 + 1,) * 2)
        opened = cv2.morphologyEx(normed, cv2.MORPH_OPEN, k_open)
        if opened.any():
            n_comp, labels = cv2.connectedComponents(opened.astype(np.uint8))
            if n_comp > 1:
                sizes   = np.bincount(labels.ravel())[1:]
                largest = int(np.argmax(sizes)) + 1
                normed  = (labels == largest).astype(np.uint8)
            else:
                normed = opened

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1,) * 2)
        normed = cv2.dilate(normed, kernel)

    return normed


def _visual_hull_carving(
    silhouettes: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    grid_size: int   = GRID_SIZE,
    distance: float  = CAMERA_DISTANCE,
    elevation: float = CAMERA_ELEVATION,
) -> np.ndarray:
    """
    Carve a dense voxel grid using normalised silhouettes and a vote threshold.

    Each silhouette is cropped to its bounding box and resized to NORM_SIZE²
    before projection, making the algorithm independent of the original image
    zoom / camera distance.  The focal length is derived analytically so the
    ±1 world extent maps to the padded boundary of the normalised image.

    A voxel is kept when it lies inside at least MIN_VOTE_FRAC of the views
    (behind-camera views abstain rather than voting against).

    Returns an (M, 3) float32 array of occupied voxel centres in [-1, 1]³.
    """
    n_views = len(silhouettes)
    sz      = float(NORM_SIZE)

    # Focal length: maps world ±1 to the inner (non-padded) boundary of the
    # normalised image at the assumed camera distance.
    # f * 1.0 / distance == (0.5 - NORM_PAD) * NORM_SIZE
    f_eff = (0.5 - NORM_PAD) * sz * distance
    cx = cy = sz / 2.0

    # Build flat array of all voxel-centre world coordinates (N, 3)
    coords    = np.linspace(-1.0, 1.0, grid_size)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    n_total = pts_world.shape[0]
    votes   = np.zeros(n_total, dtype=np.int32)
    angles  = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)

    for i, angle in enumerate(angles):
        sil = _normalize_silhouette(silhouettes[i])
        w = h = NORM_SIZE

        R, t = _build_camera(angle, elevation, distance)
        R = R.astype(np.float32)
        t = t.astype(np.float32)

        # Process in chunks to cap peak RAM at ~400 MB for 256³ grids
        for start in range(0, n_total, _PROJ_CHUNK):
            end   = min(start + _PROJ_CHUNK, n_total)
            chunk = pts_world[start:end]              # view — no allocation

            pts_cam  = chunk @ R.T + t                # (C, 3)
            z        = pts_cam[:, 2]
            in_front = z > 1e-4

            safe_z = np.where(in_front, z, 1.0)
            px = f_eff * pts_cam[:, 0] / safe_z + cx
            py = f_eff * pts_cam[:, 1] / safe_z + cy

            px_i = np.round(px).astype(np.int32)
            py_i = np.round(py).astype(np.int32)

            in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)

            in_sil = np.zeros(end - start, dtype=bool)
            idx    = np.where(in_bounds)[0]
            if idx.size:
                in_sil[idx] = sil[py_i[idx], px_i[idx]] > 0

            # Behind-camera voxels abstain (counted as passing this view)
            votes[start:end] += (in_sil | ~in_front).astype(np.int32)

    min_votes    = max(1, int(np.ceil(n_views * MIN_VOTE_FRAC)))
    occupied_3d  = (votes >= min_votes).reshape(grid_size, grid_size, grid_size)

    # ── Keep only the largest connected component ────────────────────────────
    # Satellite blobs caused by camera-model errors are discarded; only the
    # single largest contiguous region (the actual object) is retained.
    labeled, n_components = nd_label(occupied_3d)
    if n_components == 0:
        return pts_world[np.zeros(n_total, dtype=bool)]
    if n_components > 1:
        sizes         = np.bincount(labeled.ravel())[1:]   # component sizes (skip bg=0)
        largest_label = int(np.argmax(sizes)) + 1
        occupied_3d   = labeled == largest_label
        logger.info(
            "Connected components: %d found, kept largest (%d voxels), "
            "discarded %d voxels in satellite blobs",
            n_components,
            sizes[largest_label - 1],
            sizes.sum() - sizes[largest_label - 1],
        )

    return pts_world[occupied_3d.ravel()]


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
        seg_images = run.get("segmented_images", [])
        if len(seg_images) < 2:
            raise ValueError("Need at least 2 segmented images for visual hull reconstruction")

        loop = asyncio.get_event_loop()

        # Load segmented images from GridFS
        silhouettes  : list[np.ndarray]      = []
        image_sizes  : list[tuple[int, int]] = []

        for entry in seg_images:
            file_id = ObjectId(entry["segmented_file_id"])
            stream  = await fs.open_download_stream(file_id)
            data    = await stream.read()
            sil, sz = await loop.run_in_executor(None, _extract_silhouette, data)
            silhouettes.append(sil)
            image_sizes.append(sz)

        logger.info("Starting visual hull carving for run %s (%d views)", run_id, len(silhouettes))

        # Run CPU-heavy carving off the event loop
        pts = await loop.run_in_executor(
            None,
            _visual_hull_carving,
            silhouettes,
            image_sizes,
        )

        if pts.shape[0] == 0:
            raise ValueError(
                "Visual hull is empty — silhouettes may not overlap. "
                "Try more images from additional angles."
            )

        logger.info("Visual hull: %d occupied voxels for run %s", pts.shape[0], run_id)

        # Serialise point cloud
        point_list   = [{"x": -float(p[0]), "y": -float(p[1]), "z": float(p[2])} for p in pts]
        pts_bytes    = json.dumps(point_list).encode()

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
            "point_cloud_file_id": str(cloud_id),
            "point_count":         pts.shape[0],
            "method":              "visual_hull",
            "grid_size":           GRID_SIZE,
        }

        await db.runs.update_one(
            {"_id": ObjectId(run_id)},
            {"$set": {
                "status":                       "reconstructed",
                "reconstruction":               meta,
                "reconstruction_completed_at":  datetime.now(timezone.utc),
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
