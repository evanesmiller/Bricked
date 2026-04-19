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
MONGO_URI="mongodb://<host>:27017" uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

### API Endpoints

#### `POST /api/uploads/runs`
Upload images and create a new processing run.

- **Body:** `multipart/form-data` with field `images` (1–20 files, JPEG/PNG/WebP, max 50MB each)
- **Returns:** `run_id` and metadata for each uploaded image

```json
{
  "run_id": "abc123",
  "images": [
    { "file_id": "xyz789", "filename": "front.jpg", "size": 204800 }
  ]
}
```

#### `GET /api/uploads/runs/{run_id}`
Get the status and image references for a run.

```json
{
  "run_id": "abc123",
  "status": "uploaded",
  "created_at": "2026-04-18T12:00:00Z",
  "images": [...]
}
```

#### `GET /api/uploads/images/{file_id}`
Stream a stored image directly from GridFS.

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
