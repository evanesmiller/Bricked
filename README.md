# Bricked

A web app that transforms real-world objects into LEGO-style buildable models. Upload multiple photos of an object from different angles and Bricked will segment the object, reconstruct a 3D model, voxelize it, and output a LEGO brick layout with a filterable parts list you can actually build from.

## How it works

1. **Upload** – User uploads 4–16 images of an object from different angles
2. **Segment** – YOLO isolates the object from the background in each image
3. **Reconstruct** – OpenCV Structure-from-Motion estimates a coarse 3D point cloud
4. **Voxelize** – Open3D converts the 3D structure into a cube grid
5. **LEGO Convert** – Trimesh merges voxels into brick types and generates a parts list
6. **Visualize** – Three.js renders the point cloud and voxel grid interactively; the parts list shows filterable brick counts with color swatches

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend API | FastAPI |
| Database | MongoDB Atlas + GridFS |
| Python DB driver | Motor (async PyMongo) |
| Segmentation | Ultralytics YOLO |
| 3D Reconstruction | OpenCV (SfM / visual hull) |
| Voxelization | Open3D |
| LEGO Conversion | Trimesh |
| Frontend | React 19 + Vite + Tailwind CSS |
| 3D Rendering | Three.js + OrbitControls |

---

## Backend Setup

### Prerequisites

- Python 3.12 (open3d does not support 3.13+)
- MongoDB Atlas cluster (or local MongoDB 6+)

### Install dependencies

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string (use full `mongodb+srv://` URI for Atlas) |
| `MONGO_DB` | `bricked` | Database name |

### Run the server

```bash
cd backend
source venv/bin/activate
MONGO_URI="mongodb+srv://<user>:<pass>@<cluster>.mongodb.net/?retryWrites=true&w=majority" \
  uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`. Interactive API docs at `http://localhost:8000/docs`.

> **macOS + MongoDB Atlas SSL note:** The backend uses `certifi` for TLS certificate verification. This is already wired into `database.py` — no extra steps needed as long as `certifi` is installed via `requirements.txt`.

---

## Frontend Setup

### Prerequisites

- Node.js 18+

### Install dependencies

```bash
cd frontend
npm install
```

### Environment variables

Create `frontend/.env.local` if your backend runs anywhere other than `http://localhost:8000`:

```
VITE_API_BASE_URL=http://localhost:8000
```

### Run the dev server

```bash
cd frontend
npm run dev
```

App runs at `http://localhost:5173`.

### Build for production

```bash
npm run build   # output in frontend/dist/
```

---

## API Endpoints

Each run moves through a linear pipeline. Trigger each stage in order:

```
uploaded → segmented → reconstructed → voxelized → complete
```

### Upload & Storage

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/uploads/runs` | Upload 1–20 images, create a run. Body: `multipart/form-data` field `images` |
| `GET` | `/api/uploads/runs/{run_id}` | Fetch full run document from MongoDB |
| `GET` | `/api/uploads/images/{file_id}` | Stream an image from GridFS |

### Pipeline Triggers

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/runs/{run_id}/segment` | Run YOLO segmentation on uploaded images |
| `POST` | `/api/runs/{run_id}/reconstruct` | Run OpenCV SfM to build a point cloud |
| `POST` | `/api/runs/{run_id}/voxelize` | Convert point cloud to voxel grid via Open3D |
| `POST` | `/api/runs/{run_id}/lego` | Pack voxels into LEGO bricks and generate parts list |

### Model & Results

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/runs/{run_id}/status` | Poll current pipeline status |
| `GET` | `/api/runs/{run_id}/model` | Fetch full brick layout JSON |
| `GET` | `/api/runs/{run_id}/parts` | Fetch parts list (type, color, count per brick) |
| `GET` | `/api/runs/{run_id}/pointcloud` | Fetch sub-sampled point cloud for visualization |
| `GET` | `/api/runs/{run_id}/voxels` | Fetch voxel grid for visualization |

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

---

## Project Structure

```
Bricked/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI app, CORS, lifespan hooks
│   │   ├── config.py                      # Env var config (MONGO_URI, MONGO_DB)
│   │   ├── database.py                    # Motor client + GridFS bucket (certifi TLS)
│   │   ├── routers/
│   │   │   ├── uploads.py                 # Upload endpoints
│   │   │   ├── pipeline.py                # Segment / reconstruct / voxelize / lego triggers
│   │   │   └── model.py                   # Model, parts, pointcloud, voxels, status
│   │   └── services/
│   │       ├── upload_service.py          # Image storage logic
│   │       ├── segmentation_service.py    # YOLO inference
│   │       ├── reconstruction_service.py  # OpenCV SfM / visual hull
│   │       ├── voxel_service.py           # Open3D voxelization
│   │       └── lego_service.py            # Trimesh brick packing + parts list
│   └── requirements.txt
└── frontend/
    ├── public/
    │   └── pirate-ship-10061128_640.webp  # Background image
    ├── src/
    │   ├── main.jsx                       # All React components + Three.js viewers
    │   └── styles.css                     # Tailwind base + root background
    ├── package.json
    ├── vite.config.js
    └── tailwind.config.js
```
