from fastapi import APIRouter, HTTPException
from bson import ObjectId
from app.database import get_db
from app.services import segmentation_service, reconstruction_service, voxel_service, lego_service

router = APIRouter(prefix="/runs/{run_id}", tags=["pipeline"])


def _validate_object_id(run_id: str):
    try:
        return ObjectId(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")


@router.post("/segment", status_code=202)
async def segment(run_id: str):
    """Trigger YOLO segmentation on uploaded images."""
    _validate_object_id(run_id)
    return await segmentation_service.run_segmentation(run_id)


@router.post("/reconstruct", status_code=202)
async def reconstruct(run_id: str):
    """Trigger OpenCV SfM 3D reconstruction from segmented images."""
    _validate_object_id(run_id)
    return await reconstruction_service.run_reconstruction(run_id)


@router.post("/voxelize", status_code=202)
async def voxelize(run_id: str):
    """Trigger Open3D voxelization of the reconstructed point cloud."""
    _validate_object_id(run_id)
    return await voxel_service.run_voxelization(run_id)


@router.post("/lego", status_code=202)
async def convert_to_lego(run_id: str):
    """Trigger Trimesh LEGO conversion and generate the parts list."""
    _validate_object_id(run_id)
    return await lego_service.run_lego_conversion(run_id)
