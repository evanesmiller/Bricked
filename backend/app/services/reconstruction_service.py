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

from app.database import get_db, get_gridfs

logger = logging.getLogger(__name__)

# ── Carving parameters ────────────────────────────────────────────────────────

GRID_SIZE        = 64    # Voxel grid resolution (N³)
CAMERA_DISTANCE  = 3.0   # Distance from camera to object origin; object spans [-1,1]³
CAMERA_ELEVATION = 0.3   # Radians (~17°) — cameras tilt slightly downward
FOV_DEGREES      = 60.0  # Horizontal field of view assumed for every image


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

def _visual_hull_carving(
    silhouettes: list[np.ndarray],
    image_sizes: list[tuple[int, int]],
    grid_size: int    = GRID_SIZE,
    distance: float   = CAMERA_DISTANCE,
    elevation: float  = CAMERA_ELEVATION,
    fov_deg: float    = FOV_DEGREES,
) -> np.ndarray:
    """
    Carve a dense voxel grid using one silhouette per camera view.

    A voxel is kept only when it projects inside the silhouette for *every*
    view (or is behind that camera and cannot be judged).

    Returns an (M, 3) float32 array of occupied voxel centres in [-1, 1]³.
    """
    n_views = len(silhouettes)

    # Build flat array of all voxel-centre world coordinates (N, 3)
    coords = np.linspace(-1.0, 1.0, grid_size)
    gx, gy, gz = np.meshgrid(coords, coords, coords, indexing="ij")
    pts_world = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1).astype(np.float32)

    occupied = np.ones(grid_size ** 3, dtype=bool)
    angles   = np.linspace(0.0, 2.0 * np.pi, n_views, endpoint=False)

    for i, angle in enumerate(angles):
        sil = silhouettes[i]
        w, h = image_sizes[i]

        R, t = _build_camera(angle, elevation, distance)
        R = R.astype(np.float32)
        t = t.astype(np.float32)

        # Project to camera space  (N, 3)
        pts_cam = (R @ pts_world.T).T + t

        z = pts_cam[:, 2]
        in_front = z > 1e-4

        # Perspective projection → pixel coords
        f  = (w / 2.0) / np.tan(np.radians(fov_deg / 2.0))
        cx = w / 2.0
        cy = h / 2.0

        safe_z = np.where(in_front, z, 1.0)
        px = f * pts_cam[:, 0] / safe_z + cx
        py = f * pts_cam[:, 1] / safe_z + cy

        px_i = np.round(px).astype(np.int32)
        py_i = np.round(py).astype(np.int32)

        in_bounds = in_front & (px_i >= 0) & (px_i < w) & (py_i >= 0) & (py_i < h)

        in_sil = np.zeros(len(pts_world), dtype=bool)
        idx = np.where(in_bounds)[0]
        if idx.size:
            in_sil[idx] = sil[py_i[idx], px_i[idx]] > 0

        # Carve: remove voxels that are in-front-of-camera but outside silhouette
        occupied &= in_sil | ~in_front

    return pts_world[occupied]


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
        point_list   = [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2])} for p in pts]
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
