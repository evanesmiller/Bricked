# Bricked

A web app that transforms real-world objects into LEGO-style buildable models. Upload multiple photos of an object from different angles and Bricked will segment the object, reconstruct a 3D model, voxelize it, and output a LEGO brick layout with a parts list you can actually build from.

## How it works

1. **Upload** – User uploads 4-8 images of an object from different angles
2. **Segment** – YOLO isolates the object from the background in each image
3. **Reconstruct** – OpenCV Structure-from-Motion estimates a coarse 3D point cloud
4. **Voxelize** – Open3D converts the 3D structure into a cube grid
5. **LEGO Convert** – Trimesh merges voxels into brick types and generates a parts list
6. **Visualize** – Three.js renders the final LEGO model interactively in the browser

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI |
| Database | MongoDB + GridFS |
| Python DB driver | Motor (async PyMongo) |
| Segmentation | Ultralytics YOLO |
| 3D Reconstruction | OpenCV (SfM) |
| Voxelization | Open3D |
| LEGO Conversion | Trimesh |
| Frontend | React + Tailwind CSS |
| 3D Rendering | Three.js |

---

## Backend Setup

### Prerequisites

- Python 3.11+
- MongoDB instance (local or remote)

### Install dependencies

```bash
cd backend
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string |
| `MONGO_DB` | `bricked` | Database name |

### Run the server

```bash
cd backend
source venv/bin/activate      # Windows: venv\Scripts\activate
MONGO_URL="mongodb://<host>:27017" uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### API Endpoints

Each run moves through a linear pipeline. Trigger each stage in order by calling the corresponding `POST` endpoint.

```
uploaded → segmented → reconstructed → voxelized → complete
```

#### Upload & Storage

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/uploads/runs` | Upload 1–20 images, create a run. Body: `multipart/form-data` field `images` |
| `GET` | `/api/uploads/runs/{run_id}` | Fetch full run document from MongoDB |
| `GET` | `/api/uploads/images/{file_id}` | Stream an image from GridFS |

#### Pipeline Triggers

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/runs/{run_id}/segment` | Run YOLO segmentation on uploaded images |
| `POST` | `/api/runs/{run_id}/reconstruct` | Run OpenCV SfM to build a point cloud |
| `POST` | `/api/runs/{run_id}/voxelize` | Convert point cloud to voxel grid via Open3D |
| `POST` | `/api/runs/{run_id}/lego` | Pack voxels into LEGO bricks and generate parts list |

#### Model & Results (consumed by Three.js frontend)

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/runs/{run_id}/status` | Poll current pipeline status |
| `GET` | `/api/runs/{run_id}/model` | Fetch full brick layout JSON for Three.js renderer |
| `GET` | `/api/runs/{run_id}/parts` | Fetch parts list (type, color, count per brick) |

**Example `/model` response:**
```json
{
  "bricks": [
    { "x": 0, "y": 0, "z": 0, "width": 2, "depth": 4, "height": 1, "type": "2x4", "color": "#9BA19D", "color_name": "Gray" }
  ],
  "dimensions": { "width": 10, "height": 5, "depth": 8 }
}
```

**Example `/parts` response:**
```json
{
  "run_id": "abc123",
  "brick_count": 42,
  "parts": [
    { "type": "2x4", "color_name": "Gray", "color": "#9BA19D", "count": 18 },
    { "type": "1x2", "color_name": "Gray", "color": "#9BA19D", "count": 24 }
  ]
}
```

### Project structure

```
backend/
├── app/
│   ├── main.py                  # App entry point, CORS, lifespan
│   ├── config.py                # Env var config
│   ├── database.py              # Motor client + GridFS bucket
│   ├── routers/
│   │   └── uploads.py           # Upload route handlers
│   └── services/
│       └── upload_service.py    # Upload + run business logic
└── requirements.txt
```
