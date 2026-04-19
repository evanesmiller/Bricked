import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const MIN_IMAGES = 4;
const MAX_IMAGES = 8;

function formatSize(bytes) {
  if (!bytes) return "0 KB";
  const units = ["B", "KB", "MB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

// ── Point cloud Three.js renderer ────────────────────────────────────────────

function PointCloudViewer({ points }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!points?.length || !mountRef.current) return;
    const el = mountRef.current;
    const w  = el.clientWidth  || 320;
    const h  = el.clientHeight || 240;

    // ── Renderer ────────────────────────────────────────────────────────────
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    el.appendChild(renderer.domElement);

    // ── Scene / camera ───────────────────────────────────────────────────────
    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, w / h, 0.01, 100);
    camera.position.set(2, 1.5, 3);
    camera.lookAt(0, 0, 0);

    // ── Orbit controls ───────────────────────────────────────────────────────
    const controls           = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping   = true;
    controls.dampingFactor   = 0.06;
    controls.autoRotate      = true;
    controls.autoRotateSpeed = 1.2;
    // Stop auto-rotate the moment the user grabs the scene
    controls.addEventListener("start", () => { controls.autoRotate = false; });

    // ── Axes (X=red, Y=green, Z=blue) ────────────────────────────────────────
    const axes = new THREE.AxesHelper(1.5);
    scene.add(axes);

    // ── Origin sphere ─────────────────────────────────────────────────────────
    const originGeo  = new THREE.SphereGeometry(0.04, 16, 16);
    const originMat  = new THREE.MeshBasicMaterial({ color: 0xffffff });
    const originMesh = new THREE.Mesh(originGeo, originMat);
    scene.add(originMesh);

    // ── Grid on the XZ plane ─────────────────────────────────────────────────
    const grid = new THREE.GridHelper(2, 10, 0x334455, 0x223344);
    grid.position.y = -1; // sit at the bottom of the [-1,1] volume
    scene.add(grid);

    // ── Point cloud geometry ─────────────────────────────────────────────────
    const positions = new Float32Array(points.length * 3);
    const colors    = new Float32Array(points.length * 3);
    const yVals     = points.map((p) => p.y);
    const yMin      = Math.min(...yVals);
    const yMax      = Math.max(...yVals);
    const color     = new THREE.Color();

    points.forEach((p, i) => {
      positions[i * 3]     = p.x;
      positions[i * 3 + 1] = p.y;
      positions[i * 3 + 2] = p.z;

      const t = (p.y - yMin) / (yMax - yMin + 1e-6);
      color.setHSL(0.58 - t * 0.28, 0.9, 0.55); // deep-blue → cyan
      colors[i * 3]     = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color",    new THREE.BufferAttribute(colors,    3));

    const mat   = new THREE.PointsMaterial({ size: 0.05, vertexColors: true, sizeAttenuation: true });
    const cloud = new THREE.Points(geo, mat);
    scene.add(cloud);

    // ── Resize handling ───────────────────────────────────────────────────────
    const observer = new ResizeObserver(() => {
      const nw = el.clientWidth;
      const nh = el.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    });
    observer.observe(el);

    // ── Render loop ───────────────────────────────────────────────────────────
    let animId;
    const animate = () => {
      animId = requestAnimationFrame(animate);
      controls.update();
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      observer.disconnect();
      controls.dispose();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
      renderer.dispose();
      geo.dispose();
      mat.dispose();
      originGeo.dispose();
      originMat.dispose();
    };
  }, [points]);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />

      {/* Axis legend */}
      <div className="absolute bottom-2 left-2 flex flex-col gap-0.5 pointer-events-none select-none">
        <span className="flex items-center gap-1 text-xs font-mono">
          <span className="inline-block h-0.5 w-4 bg-red-500" /> X
        </span>
        <span className="flex items-center gap-1 text-xs font-mono">
          <span className="inline-block h-0.5 w-4 bg-green-500" /> Y
        </span>
        <span className="flex items-center gap-1 text-xs font-mono">
          <span className="inline-block h-0.5 w-4 bg-blue-500" /> Z
        </span>
      </div>

      {/* Interaction hint */}
      <p className="absolute top-1.5 right-2 text-xs text-slate-400 pointer-events-none select-none">
        Drag · scroll · right-drag to pan
      </p>
    </div>
  );
}

// ── Point cloud bay (replaces the static placeholder) ────────────────────────

function PointCloudBay({ pointCloud }) {
  return (
    <article className="rounded-md border border-cyan-100/15 bg-[#071d24] p-4 shadow-abyss">
      <div className="min-h-48 rounded-md border border-cyan-100/15 bg-[#031318] bg-scan-lines [background-size:24px_24px] p-1">
        {pointCloud ? (
          <div className="h-48">
            <PointCloudViewer points={pointCloud.points} />
          </div>
        ) : (
          <div className="flex h-48 items-center justify-center rounded border border-dashed border-cyan-100/20 text-center text-sm text-slate-300">
            Awaiting reconstruction
          </div>
        )}
      </div>
      <p className="mt-4 text-sm font-semibold uppercase tracking-[0.16em] text-cyan-200">
        {pointCloud ? "Reconstructed" : "Queued"}
      </p>
      <h3 className="mt-2 text-xl font-bold text-white">Point Cloud</h3>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        {pointCloud
          ? `${pointCloud.point_count.toLocaleString()} voxels via ${pointCloud.method} on a ${pointCloud.grid_size}³ grid.`
          : "Visual hull points will anchor the rough model."}
      </p>
    </article>
  );
}

function ModelBay({ title, status, copy }) {
  return (
    <article className="rounded-md border border-cyan-100/15 bg-[#071d24] p-4 shadow-abyss">
      <div className="min-h-48 rounded-md border border-cyan-100/15 bg-[#031318] bg-scan-lines [background-size:24px_24px] p-4">
        <div className="flex h-full min-h-40 items-center justify-center rounded border border-dashed border-cyan-100/20 text-center text-sm text-slate-300">
          Mesh preview placeholder
        </div>
      </div>
      <p className="mt-4 text-sm font-semibold uppercase tracking-[0.16em] text-cyan-200">{status}</p>
      <h3 className="mt-2 text-xl font-bold text-white">{title}</h3>
      <p className="mt-2 text-sm leading-6 text-slate-300">{copy}</p>
    </article>
  );
}

// ── Pipeline panel ────────────────────────────────────────────────────────────

function PipelinePanel({ run, stage, isUploading }) {
  const steps = ["Upload", "Segment", "Reconstruct", "Voxelize", "LEGO Convert"];

  function getStepStatus(index) {
    if (index === 0) {
      if (isUploading && stage === "Uploading images...") return "in-progress";
      if (run) return "completed";
    }
    if (index === 1) {
      if (isUploading && stage === "Segmenting objects...") return "in-progress";
      if (run?.segmented_images) return "completed";
    }
    if (index === 2) {
      if (isUploading && stage === "Reconstructing 3D shape...") return "in-progress";
      if (run?.reconstruction) return "completed";
    }
    return "pending";
  }

  return (
    <aside className="rounded-md border border-teal-200/15 bg-[#04181f]/85 p-5 shadow-abyss backdrop-blur">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-200">Processing Current</p>
      <h2 className="mt-2 text-2xl font-bold text-white">Pipeline placeholders are surfaced early.</h2>
      <div className="mt-6 space-y-3">
        {steps.map((step, index) => {
          const status = getStepStatus(index);
          return (
            <div key={step} className="flex items-center gap-3 rounded-md border border-cyan-100/10 bg-[#061f27] p-3">
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-sm font-black ${
                  status === "completed"
                    ? "bg-emerald-300 text-emerald-950"
                    : status === "in-progress"
                      ? "bg-amber-300 text-amber-950"
                      : "bg-cyan-950 text-cyan-100"
                }`}
              >
                {index + 1}
              </div>
              <div>
                <p className="font-semibold text-white">{step}</p>
                <p className="text-sm text-slate-300">
                  {status === "completed" ? "Completed" : status === "in-progress" ? "In Progress" : "Coming soon"}
                </p>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
  );
}

// ── Upload panel ──────────────────────────────────────────────────────────────

function UploadPanel({ canUpload, error, files, inputRef, isUploading, onFiles, onRemove, onUpload, previews, run, stage }) {
  return (
    <section className="rounded-md border border-cyan-200/20 bg-[#06202a]/85 p-5 shadow-abyss backdrop-blur">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-cyan-200">Upload Pictures</p>
          <h2 className="mt-2 text-2xl font-bold text-white">Send 4-8 object photos below deck.</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Choose JPG, PNG, or WebP images from different angles. The API stores the originals in MongoDB GridFS.
          </p>
        </div>
        <div className="rounded-md border border-teal-200/20 bg-teal-900/30 px-3 py-2 text-sm text-teal-50">
          {files.length}/{MAX_IMAGES} selected
        </div>
      </div>

      <button
        className="mt-6 flex min-h-44 w-full cursor-pointer flex-col items-center justify-center rounded-md border border-dashed border-cyan-100/45 bg-[#03151d]/80 px-5 py-8 text-center transition hover:border-cyan-100 hover:bg-[#062533] focus:outline-none focus:ring-2 focus:ring-cyan-200"
        onClick={() => inputRef.current?.click()}
        onDragOver={(event) => event.preventDefault()}
        onDrop={(event) => {
          event.preventDefault();
          onFiles(event.dataTransfer.files);
        }}
        type="button"
      >
        <span className="text-lg font-semibold text-white">Drop images here or browse</span>
        <span className="mt-2 text-sm text-slate-300">One upload creates a new processing run.</span>
      </button>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp,image/heic,image/heif,.heic,.heif"
        multiple
        className="hidden"
        onChange={(event) => { onFiles(event.target.files); event.target.value = ""; }}
      />

      {error && (
        <div className="mt-4 rounded-md border border-rose-200/40 bg-rose-950/50 px-4 py-3 text-sm text-rose-100">
          {error}
        </div>
      )}

      {previews.length > 0 && (
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {previews.map(({ file, url }, index) => (
            <figure key={`${file.name}-${file.lastModified}`} className="relative rounded-md border border-cyan-100/15 bg-[#031318] p-2">
              <button
                type="button"
                onClick={() => onRemove(index)}
                className="absolute right-1 top-1 flex h-5 w-5 items-center justify-center rounded-full bg-rose-600 text-xs font-bold text-white hover:bg-rose-400"
                aria-label="Remove image"
              >×</button>
              <img src={url} alt={file.name} className="aspect-square w-full rounded object-cover" />
              <figcaption className="mt-2 min-h-12 text-xs text-slate-300">
                <span className="block truncate font-medium text-slate-100">{file.name}</span>
                <span>{formatSize(file.size)}</span>
              </figcaption>
            </figure>
          ))}
        </div>
      )}

      <div className="mt-5 flex flex-wrap items-center gap-3">
        <button
          className="rounded-md bg-cyan-300 px-5 py-3 text-sm font-bold text-[#031318] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-slate-500 disabled:text-slate-200"
          disabled={!canUpload}
          onClick={onUpload}
          type="button"
        >
          {isUploading ? stage || "Processing..." : "Generate Lego"}
        </button>
      </div>

      {run && (
        <div className="mt-5 rounded-md border border-emerald-200/30 bg-emerald-950/35 p-4 text-sm text-emerald-50">
          <p className="font-semibold">Run created: {run.run_id}</p>
          <p className="mt-1 text-emerald-100/80">{run.images?.length ?? 0} images stored in the database.</p>
        </div>
      )}
    </section>
  );
}

// ── App root ──────────────────────────────────────────────────────────────────

function App() {
  const inputRef = useRef(null);
  const [files,       setFiles]       = useState([]);
  const [run,         setRun]         = useState(null);
  const [pointCloud,  setPointCloud]  = useState(null);
  const [error,       setError]       = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [stage,       setStage]       = useState("");

  const previews = useMemo(
    () => files.map((file) => ({ file, url: URL.createObjectURL(file) })),
    [files],
  );

  useEffect(() => {
    return () => previews.forEach((p) => URL.revokeObjectURL(p.url));
  }, [previews]);

  const canUpload = files.length >= MIN_IMAGES && files.length <= MAX_IMAGES && !isUploading;

  function updateFiles(nextFiles) {
    const incoming = Array.from(nextFiles).filter(
      (f) => f.type.startsWith("image/") || /\.(heic|heif)$/i.test(f.name),
    );
    setRun(null);
    setPointCloud(null);
    setError("");

    setFiles((prev) => {
      const existingKeys = new Set(prev.map((f) => `${f.name}-${f.size}`));
      const merged = [...prev, ...incoming.filter((f) => !existingKeys.has(`${f.name}-${f.size}`))];
      const capped = merged.slice(0, MAX_IMAGES);
      if (merged.length > MAX_IMAGES) {
        setError(`Choose ${MIN_IMAGES}–${MAX_IMAGES} images for this pass.`);
      } else if (capped.length > 0 && capped.length < MIN_IMAGES) {
        setError(`Add ${MIN_IMAGES - capped.length} more image${MIN_IMAGES - capped.length === 1 ? "" : "s"} to start.`);
      } else {
        setError("");
      }
      return capped;
    });
  }

  function removeFile(index) {
    setFiles((prev) => {
      const next = prev.filter((_, i) => i !== index);
      if (next.length > 0 && next.length < MIN_IMAGES) {
        setError(`Add ${MIN_IMAGES - next.length} more image${MIN_IMAGES - next.length === 1 ? "" : "s"} to start.`);
      } else {
        setError("");
      }
      return next;
    });
  }

  async function uploadImages() {
    if (!canUpload) return;

    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));

    setIsUploading(true);
    setError("");

    try {
      // 1 — Upload
      setStage("Uploading images...");
      const uploadRes  = await fetch(`${API_BASE_URL}/api/uploads/runs`, { method: "POST", body: formData });
      const uploadData = await uploadRes.json().catch(() => null);
      if (!uploadRes.ok) throw new Error(uploadData?.detail ?? "Upload failed.");
      setRun(uploadData);

      // 2 — Segment
      setStage("Segmenting objects...");
      const segRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/segment`, { method: "POST" });
      const segData = await segRes.json().catch(() => null);
      if (!segRes.ok) throw new Error(segData?.detail ?? "Segmentation failed.");
      setRun((prev) => ({ ...prev, ...segData }));

      // 3 — Reconstruct (visual hull)
      setStage("Reconstructing 3D shape...");
      const reconRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/reconstruct`, { method: "POST" });
      const reconData = await reconRes.json().catch(() => null);
      if (!reconRes.ok) throw new Error(reconData?.detail ?? "Reconstruction failed.");
      setRun((prev) => ({ ...prev, reconstruction: reconData }));

      // 4 — Fetch point cloud for visualization
      const pcRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/pointcloud`);
      const pcData = await pcRes.json().catch(() => null);
      if (pcRes.ok && pcData) setPointCloud(pcData);

      setStage("Done");
    } catch (err) {
      setError(err.message);
      setStage("");
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <main className="min-h-screen bg-[#041016] text-slate-50">
      <section className="relative isolate overflow-hidden">
        <img
          src="/deep-sea-banner.png"
          alt="Deep ocean with bioluminescent light"
          className="absolute inset-0 -z-20 h-full w-full object-cover opacity-60"
        />
        <div className="absolute inset-0 -z-10 bg-[linear-gradient(180deg,rgba(4,16,22,0.55),#041016_78%)]" />
        <div className="mx-auto flex min-h-[42rem] max-w-7xl flex-col px-5 py-8 sm:px-8 lg:px-10">
          <header className="flex items-center justify-between gap-4">
            <div>
              <p className="text-sm font-semibold uppercase tracking-[0.22em] text-cyan-200">Bricked</p>
              <h1 className="mt-3 max-w-4xl text-4xl font-black leading-tight text-white sm:text-6xl">
                Build a brick-ready model from a ring of photos.
              </h1>
            </div>
            <div className="hidden rounded-md border border-cyan-200/30 bg-cyan-950/40 px-4 py-3 text-sm text-cyan-50 backdrop-blur md:block">
              API: {API_BASE_URL}
            </div>
          </header>

          <div className="mt-10 grid flex-1 gap-6 lg:grid-cols-[1.05fr_0.95fr]">
            <UploadPanel
              canUpload={canUpload}
              error={error}
              files={files}
              inputRef={inputRef}
              isUploading={isUploading}
              onFiles={updateFiles}
              onRemove={removeFile}
              onUpload={uploadImages}
              previews={previews}
              run={run}
              stage={stage}
            />
            <PipelinePanel run={run} stage={stage} isUploading={isUploading} />
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 pb-10 sm:px-8 lg:grid-cols-[1fr_1fr_1fr] lg:px-10">
        <ModelBay
          title="Segmented Views"
          status="Waiting for upload"
          copy="Object masks from each image will surface here."
        />
        <PointCloudBay pointCloud={pointCloud} />
        <ModelBay
          title="Mesh Draft"
          status="Queued"
          copy="The reconstructed mesh preview will land in this bay."
        />
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-12 sm:px-8 lg:px-10">
        <div className="grid gap-5 rounded-md border border-teal-200/15 bg-[#071d24] p-5 shadow-abyss lg:grid-cols-[0.8fr_1.2fr]">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-200">Final Voxel Model</p>
            <h2 className="mt-3 text-2xl font-bold text-white">Voxel bay awaiting reconstruction.</h2>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              The final LEGO-style cube grid will appear here once segmentation, mesh reconstruction, voxelization,
              and brick conversion are connected.
            </p>
          </div>
          <div className="min-h-72 rounded-md border border-cyan-200/20 bg-[#031318] bg-scan-lines [background-size:28px_28px] p-4">
            <div className="grid h-full grid-cols-8 grid-rows-5 gap-2">
              {Array.from({ length: 40 }).map((_, index) => (
                <div
                  key={index}
                  className={`rounded-sm border ${
                    index % 7 === 0
                      ? "border-rose-200/50 bg-rose-300/20"
                      : index % 4 === 0
                        ? "border-cyan-200/50 bg-cyan-300/20"
                        : "border-teal-200/25 bg-teal-300/10"
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
