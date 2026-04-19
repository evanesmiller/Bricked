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
# Rule: NORM_SIZE = 2 × GRID_SIZE so adjacent voxels project to distinct pixels.
NORM_SIZE        = 512   # Normalised silhouette side length in pixels
NORM_PAD         = 0.10  # Fractional padding added around the bounding box
DILATION_PX      = 2     # Extra dilation on the normalised mask (error margin)
THIN_OPEN_PX     = 3     # Opening kernel radius to disconnect thin protrusions (straws, stems)

# Concavity-enhanced carving: voxels projecting into the "phantom" zone
# (inside silhouette convex hull but outside the silhouette itself) are
# counted as concavity hits. If a voxel accumulates this many hits across
# all views it is carved regardless of the vote threshold.
CONCAVITY_VETO   = 2

# Coords are generated lazily per chunk — no large pts_world allocation.
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
    orig_bgr: np.ndarray | None = None,
    out_size: int    = NORM_SIZE,
    pad_frac: float  = NORM_PAD,
    dilation_px: int = DILATION_PX,
    open_px: int     = THIN_OPEN_PX,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Crop to the object bounding box, add padding, resize to a square, apply
    optional opening to remove thin protrusions, then dilate for tolerance.

    If orig_bgr is provided (same H×W as mask), applies the identical crop and
    resize to produce a normalised RGB colour image for colour sampling.

    Returns (silhouette, concavity_mask, normed_rgb_or_None).
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        empty = np.zeros((out_size, out_size), dtype=np.uint8)
        return empty, empty, None

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
        empty = np.zeros((out_size, out_size), dtype=np.uint8)
        return empty, empty, None

    normed = cv2.resize(cropped, (out_size, out_size), interpolation=cv2.INTER_NEAREST)

    # Normalise colour image with the same crop before any morphological ops
    normed_rgb: np.ndarray | None = None
    if orig_bgr is not None:
        color_crop = orig_bgr[r0:r1, c0:c1]
        if 0 not in color_crop.shape:
            color_resized = cv2.resize(color_crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
            normed_rgb = cv2.cvtColor(color_resized, cv2.COLOR_BGR2RGB)

    # Remove thin protrusions (straws, stems) via opening + largest component
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

    # Compute concavity mask BEFORE dilation so the hull boundary is clean.
    # Concavity = pixels inside the silhouette's convex hull but outside the
    # silhouette itself — the "phantom" regions visual hull keeps by default
    # (e.g. the space between trunk and face, between arm and body).
    concavity = np.zeros_like(normed)
    contours, _ = cv2.findContours(normed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull_mask = np.zeros_like(normed)
        for cnt in contours:
            hull_pts = cv2.convexHull(cnt)
            cv2.fillConvexPoly(hull_mask, hull_pts, 1)
        concavity = ((hull_mask > 0) & (normed == 0)).astype(np.uint8)

    if dilation_px > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_px * 2 + 1,) * 2)
        normed = cv2.dilate(normed, kernel)

    return normed, concavity, normed_rgb


def _sample_colors(
    occupied_pts: np.ndarray,
    view_data: list,
    f_eff: float,
    cx: float,
    cy: float,
    w: int,
    h: int,
) -> np.ndarray:
    """
    Assign each occupied voxel the colour from the **nearest valid camera view**.

    For each voxel, iterate over all views that have a colour image and where the
    voxel projects inside the silhouette.  Keep only the sample from the view with
    the smallest camera-space z-depth (i.e. the camera most directly facing that
    voxel).  This prevents cross-view colour bleed: a white-front voxel picks the
    front camera (smallest z from front), not an average with the pink back.

    After sampling, saturation is boosted so that studio/diffuse lighting doesn't
    wash colours out to near-gray.

    Returns (M, 3) uint8 RGB array; voxels with no valid sample default to gray.
    """
    M = len(occupied_pts)
    views_with_color = [v for v in view_data if v[4] is not None]

    if not views_with_color:
        logger.warning("_sample_colors: no colour images available — returning gray")
        return np.full((M, 3), 128, dtype=np.uint8)

    best_z     = np.full(M, np.inf, dtype=np.float32)
    best_color = np.full((M, 3), np.nan, dtype=np.float32)

    for sil, _cav, R, t, normed_rgb in views_with_color:
        pts_cam  = occupied_pts @ R.T + t
        z        = pts_cam[:, 2]
        in_front = z > 1e-4

        safe_z = np.where(in_front, z, 1.0)
        px = f_eff * pts_cam[:, 0] / safe_z + cx
        py = f_eff * pts_cam[:, 1] / safe_z + cy

        px_i = np.round(px).astype(np.int32)
        py_i = np.round(py).astype(np.int32)

        in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)
        valid     = np.where(in_bounds)[0]

        if not valid.size:
            continue

        in_sil  = sil[py_i[valid], px_i[valid]] > 0
        sil_idx = valid[in_sil]
        if not sil_idx.size:
            continue

        # Update only where this view is closer (smaller z-depth) than any prior
        closer = z[sil_idx] < best_z[sil_idx]
        upd    = sil_idx[closer]
        if upd.size:
            best_z[upd]     = z[upd]
            best_color[upd] = normed_rgb[py_i[upd], px_i[upd]].astype(np.float32)

    sampled = (~np.isnan(best_color).any(axis=1)).sum()
    logger.info("_sample_colors: %d / %d voxels got a colour sample", sampled, M)

    no_data = np.isnan(best_color).any(axis=1)
    best_color[no_data] = 128.0
    colors = np.clip(best_color, 0, 255).astype(np.uint8)

    # ── Saturation boost — recovers vivid colours washed out by diffuse lighting
    hsv = cv2.cvtColor(colors.reshape(1, M, 3), cv2.COLOR_RGB2HSV).reshape(M, 3).astype(np.float32)
    hsv[:, 1] = np.clip(hsv[:, 1] * 1.8 + 25, 0, 255)   # multiply + floor lift
    colors = cv2.cvtColor(hsv.astype(np.uint8).reshape(1, M, 3), cv2.COLOR_HSV2RGB).reshape(M, 3)

    return colors


def _visual_hull_carving(
    silhouettes: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    orig_images: list[np.ndarray] | None = None,
    grid_size: int   = GRID_SIZE,
    distance: float  = CAMERA_DISTANCE,
    elevation: float = CAMERA_ELEVATION,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Carve a dense voxel grid using normalised silhouettes and a vote threshold,
    enhanced with concavity-based veto carving.

    If orig_images (list of BGR arrays, same order as silhouettes) is provided,
    a second colour-sampling pass runs over occupied voxels only and returns
    per-voxel average RGB sampled from the original photographs.

    Returns:
        pts    (M, 3) float32 — world coordinates of occupied voxel centres
        colors (M, 3) uint8  — RGB colour per voxel (gray if no images given)
    """
    n_views = len(silhouettes)
    sz      = float(NORM_SIZE)
    f_eff   = (0.5 - NORM_PAD) * sz * distance
    cx = cy = sz / 2.0
    w  = h  = NORM_SIZE

    coords  = np.linspace(-1.0, 1.0, grid_size, dtype=np.float32)
    n_total = grid_size ** 3
    votes   = np.zeros(n_total, dtype=np.int32)
    concav  = np.zeros(n_total, dtype=np.int16)
    angles  = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)

    # Pre-compute silhouettes, concavity masks, camera matrices, and colour images
    view_data: list[tuple] = []
    for i, angle in enumerate(angles):
        orig_bgr = orig_images[i] if orig_images else None
        sil, cav, normed_rgb = _normalize_silhouette(silhouettes[i], orig_bgr=orig_bgr)
        R, t = _build_camera(angle, elevation, distance)
        view_data.append((sil, cav, R.astype(np.float32), t.astype(np.float32), normed_rgb))

    gs2 = grid_size * grid_size

    for chunk_start in range(0, n_total, _PROJ_CHUNK):
        chunk_end = min(chunk_start + _PROJ_CHUNK, n_total)

        flat_idx = np.arange(chunk_start, chunk_end, dtype=np.int32)
        ix = flat_idx // gs2
        iy = (flat_idx // grid_size) % grid_size
        iz = flat_idx % grid_size
        chunk = np.stack([coords[ix], coords[iy], coords[iz]], axis=1)
        del flat_idx, ix, iy, iz

        chunk_votes = np.zeros(chunk_end - chunk_start, dtype=np.int32)
        chunk_concav = np.zeros(chunk_end - chunk_start, dtype=np.int16)

        for sil, cav, R, t, _normed_rgb in view_data:
            pts_cam  = chunk @ R.T + t
            z        = pts_cam[:, 2]
            in_front = z > 1e-4

            safe_z = np.where(in_front, z, 1.0)
            px = f_eff * pts_cam[:, 0] / safe_z + cx
            py = f_eff * pts_cam[:, 1] / safe_z + cy

            px_i = np.round(px).astype(np.int32)
            py_i = np.round(py).astype(np.int32)

            in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)
            valid     = np.where(in_bounds)[0]

            in_sil = np.zeros(chunk_end - chunk_start, dtype=bool)
            in_cav = np.zeros(chunk_end - chunk_start, dtype=bool)
            if valid.size:
                in_sil[valid] = sil[py_i[valid], px_i[valid]] > 0
                in_cav[valid] = cav[py_i[valid], px_i[valid]] > 0

            chunk_votes  += (in_sil | ~in_front).astype(np.int32)
            chunk_concav += (in_cav & in_front).astype(np.int16)

        votes[chunk_start:chunk_end]  = chunk_votes
        concav[chunk_start:chunk_end] = chunk_concav

    min_votes   = max(1, int(np.ceil(n_views * MIN_VOTE_FRAC)))
    occupied_3d = (
        (votes >= min_votes) & (concav < CONCAVITY_VETO)
    ).reshape(grid_size, grid_size, grid_size)
    del votes, concav

    # ── Keep only the largest connected component ────────────────────────────
    labeled, n_components = nd_label(occupied_3d)
    del occupied_3d
    if n_components == 0:
        empty_pts = np.empty((0, 3), dtype=np.float32)
        empty_clr = np.empty((0, 3), dtype=np.uint8)
        return empty_pts, empty_clr
    if n_components > 1:
        sizes         = np.bincount(labeled.ravel())[1:]
        largest_label = int(np.argmax(sizes)) + 1
        keep          = labeled == largest_label
        logger.info(
            "Connected components: %d found, kept largest (%d voxels), "
            "discarded %d voxels in satellite blobs",
            n_components,
            sizes[largest_label - 1],
            sizes.sum() - sizes[largest_label - 1],
        )
    else:
        keep = labeled > 0
    del labeled

    # Reconstruct world coordinates for occupied voxels
    occ_idx = np.where(keep.ravel())[0].astype(np.int32)
    del keep
    ix = occ_idx // gs2
    iy = (occ_idx // grid_size) % grid_size
    iz = occ_idx % grid_size
    occupied_pts = np.stack([coords[ix], coords[iy], coords[iz]], axis=1)

    # ── Colour sampling pass (only over surviving voxels — very fast) ────────
    colors = _sample_colors(occupied_pts, view_data, f_eff, cx, cy, w, h)

    return occupied_pts, colors


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

        silhouettes : list[np.ndarray]      = []
        image_sizes : list[tuple[int, int]] = []
        orig_images : list[np.ndarray]      = []

        for entry in seg_images:
            # Load segmented mask
            seg_stream = await fs.open_download_stream(ObjectId(entry["segmented_file_id"]))
            seg_data   = await seg_stream.read()
            sil, sz    = await loop.run_in_executor(None, _extract_silhouette, seg_data)
            silhouettes.append(sil)
            image_sizes.append(sz)

            # Load original colour image for colour sampling
            orig_stream = await fs.open_download_stream(ObjectId(entry["original_file_id"]))
            orig_data   = await orig_stream.read()
            buf         = np.frombuffer(orig_data, dtype=np.uint8)
            bgr         = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                # cv2 can't decode HEIC/HEIF — fall back to Pillow
                try:
                    from PIL import Image as _PILImage
                    pil_img = _PILImage.open(io.BytesIO(orig_data)).convert("RGB")
                    bgr     = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    bgr = None
            if bgr is None:
                logger.warning("Could not decode original image for %s — colours will be gray", entry.get("filename", "?"))
            orig_images.append(bgr)

        logger.info("Starting visual hull carving for run %s (%d views)", run_id, len(silhouettes))

        pts, colors = await loop.run_in_executor(
            None,
            _visual_hull_carving,
            silhouettes,
            image_sizes,
            orig_images,
        )

        if pts.shape[0] == 0:
            raise ValueError(
                "Visual hull is empty — silhouettes may not overlap. "
                "Try more images from additional angles."
            )

        logger.info("Visual hull: %d occupied voxels for run %s", pts.shape[0], run_id)

        # Serialise point cloud — 180° rotation around Z (negate X and Y)
        point_list = [
            {
                "x": -float(p[0]), "y": -float(p[1]), "z": float(p[2]),
                "r": int(c[0]),    "g": int(c[1]),    "b": int(c[2]),
            }
            for p, c in zip(pts, colors)
        ]
        pts_bytes = json.dumps(point_list).encode()

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
