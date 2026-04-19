"""
Model serving endpoints — consumed by the Three.js frontend renderer.
"""
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from bson import ObjectId
from app.database import get_db, get_gridfs

router = APIRouter(prefix="/runs/{run_id}", tags=["model"])


def _validate_object_id(run_id: str) -> ObjectId:
    try:
        return ObjectId(run_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_id")


@router.get("/model")
async def get_model(run_id: str):
    """
    Return the LEGO brick layout as JSON for Three.js rendering.

    Response shape:
    {
      "bricks": [{"x","y","z","width","depth","height","type","color","color_name"}, ...],
      "dimensions": {"width", "height", "depth"}
    }
    """
    oid = _validate_object_id(run_id)
    db = get_db()
    gridfs = get_gridfs()

    run = await db.runs.find_one({"_id": oid})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Model not ready — current status: '{run.get('status')}'",
        )

    model_file_id = run["lego"]["model_file_id"]
    stream = await gridfs.open_download_stream(ObjectId(model_file_id))
    data = json.loads(await stream.read())
    return JSONResponse(content=data)


@router.get("/parts")
async def get_parts(run_id: str):
    """
    Return the LEGO parts list for the completed run.

    Response shape:
    {
      "run_id": "...",
      "brick_count": 42,
      "parts": [{"type","color_name","color","count"}, ...]
    }
    """
    oid = _validate_object_id(run_id)
    db = get_db()

    run = await db.runs.find_one({"_id": oid})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Parts list not ready — current status: '{run.get('status')}'",
        )

    lego = run["lego"]
    return {
        "run_id": run_id,
        "brick_count": lego["brick_count"],
        "parts": lego["parts_list"],
    }


@router.get("/status")
async def get_status(run_id: str):
    """
    Poll the current pipeline status for a run.

    Possible statuses:
      uploaded → segmenting → segmented → reconstructing → reconstructed
      → voxelizing → voxelized → converting → complete | failed
    """
    oid = _validate_object_id(run_id)
    db = get_db()

    run = await db.runs.find_one({"_id": oid}, {"status": 1, "error": 1})
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {"run_id": run_id, "status": run["status"], "error": run.get("error")}
