import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const MIN_IMAGES = 8;
const MAX_IMAGES = 16;

function formatSize(bytes) {
  if (!bytes) return "0 KB";
  const units = ["B", "KB", "MB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

// ── Shared Three.js scene factory ─────────────────────────────────────────────

function makeScene(el) {
  const w = el.clientWidth || 400;
  const h = el.clientHeight || 300;
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(w, h);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  el.appendChild(renderer.domElement);

  const scene  = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(55, w / h, 0.01, 2000);

  const controls           = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping   = true;
  controls.dampingFactor   = 0.06;
  controls.autoRotate      = true;
  controls.autoRotateSpeed = 1.2;
  controls.addEventListener("start", () => { controls.autoRotate = false; });

  const observer = new ResizeObserver(() => {
    const nw = el.clientWidth, nh = el.clientHeight;
    camera.aspect = nw / nh;
    camera.updateProjectionMatrix();
    renderer.setSize(nw, nh);
  });
  observer.observe(el);

  let animId;
  const animate = () => {
    animId = requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };
  animate();

  const dispose = (extras = []) => {
    cancelAnimationFrame(animId);
    observer.disconnect();
    controls.dispose();
    if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    renderer.dispose();
    extras.forEach((d) => d?.dispose?.());
  };

  return { scene, camera, controls, renderer, dispose };
}

// ── Point cloud Three.js renderer ────────────────────────────────────────────

function PointCloudViewer({ points }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!points?.length || !mountRef.current) return;
    const el = mountRef.current;

    const { scene, camera, controls, dispose } = makeScene(el);
    camera.position.set(2, 1.5, 3);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AxesHelper(1.5));
    const originMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.04, 16, 16),
      new THREE.MeshBasicMaterial({ color: 0xffffff }),
    );
    scene.add(originMesh);
    const grid = new THREE.GridHelper(2, 10, 0x334455, 0x223344);
    grid.position.y = -1;
    scene.add(grid);

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
      if (p.r !== undefined) {
        color.setRGB(p.r / 255, p.g / 255, p.b / 255);
      } else {
        const t = (p.y - yMin) / (yMax - yMin + 1e-6);
        color.setHSL(0.58 - t * 0.28, 0.9, 0.55);
      }
      colors[i * 3]     = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });

    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setAttribute("color",    new THREE.BufferAttribute(colors,    3));
    const mat   = new THREE.PointsMaterial({ size: 0.02, vertexColors: true, sizeAttenuation: true });
    scene.add(new THREE.Points(geo, mat));

    return () => dispose([geo, mat]);
  }, [points]);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
      <div className="absolute bottom-2 left-2 flex flex-col gap-0.5 pointer-events-none select-none">
        <span className="flex items-center gap-1 text-xs font-mono"><span className="inline-block h-0.5 w-4 bg-red-500" /> X</span>
        <span className="flex items-center gap-1 text-xs font-mono"><span className="inline-block h-0.5 w-4 bg-green-500" /> Y</span>
        <span className="flex items-center gap-1 text-xs font-mono"><span className="inline-block h-0.5 w-4 bg-blue-500" /> Z</span>
      </div>
      <p className="absolute top-1.5 right-2 text-xs text-slate-400 pointer-events-none select-none">
        Drag · scroll · right-drag to pan
      </p>
    </div>
  );
}

// ── Voxel grid Three.js renderer ─────────────────────────────────────────────

function VoxelGridViewer({ voxels }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!voxels?.length || !mountRef.current) return;
    const el = mountRef.current;

    const xs = voxels.map((v) => v.x), ys = voxels.map((v) => v.y), zs = voxels.map((v) => v.z);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const zMin = Math.min(...zs), zMax = Math.max(...zs);
    const cx = (xMin + xMax) / 2, cy = (yMin + yMax) / 2, cz = (zMin + zMax) / 2;
    const span = Math.max(xMax - xMin, yMax - yMin, zMax - zMin, 1);

    const { scene, camera, controls, dispose } = makeScene(el);
    camera.position.set(cx + span * 1.2, cy + span * 0.8, cz + span * 1.5);
    camera.lookAt(cx, cy, cz);
    camera.far = span * 20;
    camera.updateProjectionMatrix();
    controls.target.set(cx, cy, cz);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dir = new THREE.DirectionalLight(0xffffff, 0.8);
    dir.position.set(cx + span, cy + span * 2, cz + span);
    scene.add(dir);

    const grid = new THREE.GridHelper(span * 2.5, 12, 0x334455, 0x223344);
    grid.position.set(cx, yMin - 0.5, cz);
    scene.add(grid);

    const geo   = new THREE.BoxGeometry(0.85, 0.85, 0.85);
    const mat   = new THREE.MeshLambertMaterial();
    const mesh  = new THREE.InstancedMesh(geo, mat, voxels.length);
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    voxels.forEach((v, i) => {
      dummy.position.set(v.x, v.y, v.z);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      if (v.r !== undefined) {
        color.setRGB(v.r / 255, v.g / 255, v.b / 255);
      } else {
        const t = (v.y - yMin) / (yMax - yMin + 1e-6);
        color.setHSL(0.58 - t * 0.28, 0.9, 0.55);
      }
      mesh.setColorAt(i, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);

    return () => dispose([geo, mat]);
  }, [voxels]);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
      <p className="absolute top-1.5 right-2 text-xs text-slate-400 pointer-events-none select-none">
        Drag · scroll · right-drag to pan
      </p>
    </div>
  );
}

// ── LEGO model Three.js renderer ─────────────────────────────────────────────

function LegoModelViewer({ model }) {
  const mountRef = useRef(null);

  useEffect(() => {
    if (!model?.bricks?.length || !mountRef.current) return;
    const el = mountRef.current;

    const { bricks, dimensions } = model;
    const cx = dimensions.width  / 2;
    const cy = dimensions.height / 2;
    const cz = dimensions.depth  / 2;
    const span = Math.max(dimensions.width, dimensions.height, dimensions.depth, 1);

    const { scene, camera, controls, dispose } = makeScene(el);
    camera.position.set(cx + span * 1.5, cy + span, cz + span * 2);
    camera.lookAt(cx, cy, cz);
    camera.far = span * 30;
    camera.updateProjectionMatrix();
    controls.target.set(cx, cy, cz);
    controls.autoRotateSpeed = 0.8;

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const dir1 = new THREE.DirectionalLight(0xffffff, 1.0);
    dir1.position.set(cx + span, cy + span * 2, cz + span);
    scene.add(dir1);
    const dir2 = new THREE.DirectionalLight(0xffffff, 0.3);
    dir2.position.set(cx - span, cy + span, cz - span);
    scene.add(dir2);

    const grid = new THREE.GridHelper(span * 2.5, 12, 0x334455, 0x223344);
    grid.position.set(cx, -0.5, cz);
    scene.add(grid);

    // Single InstancedMesh — dummy scale encodes each brick's footprint
    const geo   = new THREE.BoxGeometry(1, 1, 1);
    const mat   = new THREE.MeshLambertMaterial();
    const mesh  = new THREE.InstancedMesh(geo, mat, bricks.length);
    const dummy = new THREE.Object3D();
    const color = new THREE.Color();

    bricks.forEach((b, i) => {
      dummy.position.set(b.x + b.width / 2, b.y + b.height / 2, b.z + b.depth / 2);
      dummy.scale.set(b.width * 0.94, b.height * 0.94, b.depth * 0.94);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      color.set(b.color);
      mesh.setColorAt(i, color);
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    scene.add(mesh);

    return () => dispose([geo, mat]);
  }, [model]);

  return (
    <div className="relative w-full h-full">
      <div ref={mountRef} className="w-full h-full" />
      <p className="absolute top-1.5 right-2 text-xs text-slate-400 pointer-events-none select-none">
        Drag · scroll · right-drag to pan
      </p>
    </div>
  );
}

// ── Segmented views bay ───────────────────────────────────────────────────────

function FillPill({ value }) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 75 ? "bg-emerald-500 text-white"
    : pct >= 50 ? "bg-amber-400 text-amber-950"
    :             "bg-rose-500 text-white";
  return (
    <span
      className={`absolute top-1 right-1 rounded px-1.5 py-0.5 text-[10px] font-bold leading-none ${color}`}
      title={`Mask fill ratio: ${pct}% of bounding box covered`}
    >
      {pct}%
    </span>
  );
}

function SegmentedViewsBay({ run }) {
  const segmented = run?.segmented_images ?? [];
  const skipped   = run?.skipped_images   ?? [];
  const done      = segmented.length > 0;

  return (
    <article className="rounded-md border border-cyan-100/15 bg-[#071d24] p-4 shadow-abyss">
      <div className="rounded-md border border-cyan-100/15 bg-[#031318] bg-scan-lines [background-size:24px_24px] p-2">
        {done ? (
          <div className="grid max-h-56 grid-cols-2 gap-1.5 overflow-y-auto pr-0.5 sm:grid-cols-3">
            {segmented.map((img) => (
              <figure
                key={img.segmented_file_id}
                className="relative overflow-hidden rounded border border-cyan-100/10 bg-[#020e13]"
              >
                <img
                  src={`${API_BASE_URL}/api/uploads/images/${img.segmented_file_id}`}
                  alt={img.filename}
                  className="aspect-square w-full object-contain"
                  loading="lazy"
                />
                <FillPill value={img.detection.fill_ratio} />
              </figure>
            ))}
          </div>
        ) : (
          <div className="flex h-48 items-center justify-center rounded border border-dashed border-cyan-100/20 text-center text-sm text-slate-300">
            {run ? "No images passed segmentation" : "Awaiting segmentation"}
          </div>
        )}
      </div>

      <p className="mt-4 text-sm font-semibold uppercase tracking-[0.16em] text-cyan-200">
        {done ? "Segmented" : "Waiting for upload"}
      </p>
      <h3 className="mt-2 text-xl font-bold text-white">Segmented Views</h3>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        {done
          ? `${segmented.length} mask${segmented.length === 1 ? "" : "s"} extracted. Badge shows mask fill ratio.`
          : "Object masks from each image will surface here."}
        {skipped.length > 0 && ` ${skipped.length} image${skipped.length === 1 ? "" : "s"} skipped.`}
      </p>

      {skipped.length > 0 && (
        <div className="mt-3 rounded border border-amber-200/20 bg-amber-950/25 px-3 py-2">
          <p className="text-xs font-semibold text-amber-300">Skipped — poor mask quality or no detection</p>
          <ul className="mt-1 space-y-0.5">
            {skipped.map((s) => (
              <li key={s.filename} className="truncate text-xs text-amber-200/70" title={s.reason}>
                {s.filename}
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}

// ── Point cloud bay ───────────────────────────────────────────────────────────

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

// ── Voxel grid bay ────────────────────────────────────────────────────────────

function VoxelBay({ voxelData }) {
  return (
    <article className="rounded-md border border-cyan-100/15 bg-[#071d24] p-4 shadow-abyss">
      <div className="min-h-48 rounded-md border border-cyan-100/15 bg-[#031318] bg-scan-lines [background-size:24px_24px] p-4">
        <div className="flex h-full min-h-40 items-center justify-center rounded border border-dashed border-cyan-100/20 text-center text-sm text-slate-300">
          Awaiting voxelization
        </div>
      </div>
      <p className="mt-4 text-sm font-semibold uppercase tracking-[0.16em] text-cyan-200">
        {voxelData ? "Voxelized" : "Queued"}
      </p>
      <h3 className="mt-2 text-xl font-bold text-white">Voxel Grid</h3>
      <p className="mt-2 text-sm leading-6 text-slate-300">
        {voxelData
          ? `${voxelData.voxel_count.toLocaleString()} voxels at ${voxelData.voxel_size} unit resolution.`
          : "Open3D converts the point cloud into a discrete cube grid."}
      </p>
    </article>
  );
}

// ── LEGO model bay ────────────────────────────────────────────────────────────

function LegoBay({ legoModel, run }) {
  const parts = run?.lego?.parts_list ?? [];

  return (
    <section className="mx-auto max-w-7xl px-5 pb-12 sm:px-8 lg:px-10">
      <div className="grid gap-5 rounded-md border border-teal-200/15 bg-[#071d24] p-5 shadow-abyss lg:grid-cols-[0.8fr_1.2fr]">
        <div>
          <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-200">Final LEGO Model</p>
          <h2 className="mt-3 text-2xl font-bold text-white">
            {legoModel
              ? `${legoModel.bricks.length.toLocaleString()} bricks assembled.`
              : "LEGO conversion queued."}
          </h2>
          <p className="mt-3 text-sm leading-6 text-slate-300">
            {legoModel
              ? `${legoModel.dimensions.width}×${legoModel.dimensions.height}×${legoModel.dimensions.depth} studs.`
              : "Greedy brick packing runs layer-by-layer after voxelization."}
          </p>

          {parts.length > 0 && (
            <div className="mt-4 max-h-52 overflow-y-auto rounded border border-teal-200/20 bg-teal-950/30 p-3">
              <p className="text-xs font-semibold uppercase tracking-wider text-teal-300">Parts List</p>
              <ul className="mt-2 space-y-1">
                {parts.map((p) => (
                  <li key={`${p.type}-${p.color_name}`} className="flex items-center gap-2 text-xs text-slate-300">
                    <span
                      className="inline-block h-3 w-3 shrink-0 rounded-sm border border-white/20"
                      style={{ backgroundColor: p.color }}
                    />
                    <span className="font-medium text-slate-100">{p.type}</span>
                    <span className="text-slate-400">{p.color_name}</span>
                    <span className="ml-auto font-mono text-teal-200">×{p.count}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="min-h-72 rounded-md border border-cyan-200/20 bg-[#031318] bg-scan-lines [background-size:28px_28px] p-1">
          {legoModel ? (
            <div className="h-72">
              <LegoModelViewer model={legoModel} />
            </div>
          ) : (
            <div className="grid h-full grid-cols-8 grid-rows-5 gap-2 p-3">
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
          )}
        </div>
      </div>
    </section>
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
    if (index === 3) {
      if (isUploading && stage === "Voxelizing...") return "in-progress";
      if (run?.voxelization) return "completed";
    }
    if (index === 4) {
      if (isUploading && stage === "Converting to LEGO...") return "in-progress";
      if (run?.lego) return "completed";
    }
    return "pending";
  }

  return (
    <aside className="rounded-md border border-teal-200/15 bg-[#04181f]/85 p-5 shadow-abyss backdrop-blur">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-200">Processing Current</p>
      <h2 className="mt-2 text-2xl font-bold text-white">Processing pipeline beneath the waves.</h2>
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
                  {status === "completed" ? "Completed" : status === "in-progress" ? "In Progress" : "Pending"}
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
          <h2 className="mt-2 text-2xl font-bold text-white">Stow away 8–16 images o’ yer object below deck!</h2>
          <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-300">
            Hoist yer camera, matey, and gather yer images in JPG, PNG, HEIC, or WebP from all around the booty! For the finest plunder, snap 8 shots level with the object, turnin’ it 45° each time, then climb above and take 8 more from on high at the same bearings.
          </p>
        </div>
        <div className="rounded-md border border-teal-200/20 bg-teal-900/30 px-5 py-3 text-sm font-bold text-teal-50">
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
      </button>

      <button
        className="mt-3 w-full rounded-md bg-cyan-300 px-5 py-3 text-sm font-bold text-[#031318] transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:bg-slate-500 disabled:text-slate-200"
        disabled={!canUpload}
        onClick={onUpload}
        type="button"
      >
        {isUploading ? stage || "Processing..." : "Generate Lego"}
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
  const [files,      setFiles]      = useState([]);
  const [run,        setRun]        = useState(null);
  const [pointCloud, setPointCloud] = useState(null);
  const [voxelData,  setVoxelData]  = useState(null);
  const [legoModel,  setLegoModel]  = useState(null);
  const [error,      setError]      = useState("");
  const [isUploading, setIsUploading] = useState(false);
  const [stage,      setStage]      = useState("");

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
    setVoxelData(null);
    setLegoModel(null);
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

      // 3 — Reconstruct (COLMAP SfM → point cloud)
      setStage("Reconstructing 3D shape...");
      const reconRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/reconstruct`, { method: "POST" });
      const reconData = await reconRes.json().catch(() => null);
      if (!reconRes.ok) throw new Error(reconData?.detail ?? "Reconstruction failed.");
      setRun((prev) => ({ ...prev, reconstruction: reconData }));

      // Fetch point cloud for visualization (non-blocking on error)
      const pcRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/pointcloud`);
      const pcData = await pcRes.json().catch(() => null);
      if (pcRes.ok && pcData) setPointCloud(pcData);

      // 5 — Voxelize
      setStage("Voxelizing...");
      const voxRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/voxelize`, { method: "POST" });
      const voxData = await voxRes.json().catch(() => null);
      if (!voxRes.ok) throw new Error(voxData?.detail ?? "Voxelization failed.");
      setRun((prev) => ({ ...prev, voxelization: voxData }));

      // 6 — LEGO conversion
      setStage("Converting to LEGO...");
      const legoRes  = await fetch(`${API_BASE_URL}/api/runs/${uploadData.run_id}/lego`, { method: "POST" });
      const legoData = await legoRes.json().catch(() => null);
      if (!legoRes.ok) throw new Error(legoData?.detail ?? "LEGO conversion failed.");
      setRun((prev) => ({ ...prev, lego: legoData }));

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
                Flick it up & Brick it up
              </h1>
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
        <SegmentedViewsBay run={run} />
        <PointCloudBay pointCloud={pointCloud} />
        <VoxelBay voxelData={voxelData} />
      </section>

      <section className="mx-auto max-w-7xl px-5 pb-12 sm:px-8 lg:px-10">
        <div className="grid gap-5 rounded-md border border-teal-200/15 bg-[#071d24] p-5 shadow-abyss lg:grid-cols-[0.8fr_1.2fr]">
          <div>
            <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-200">Final Voxel Model</p>
            <h2 className="mt-3 text-2xl font-bold text-white">Voxel bay awaiting reconstruction.</h2>
            <p className="mt-3 text-sm leading-6 text-slate-300">
              Arrr, the final LEGO-style cube grid be showin’ itself here once ye’ve lashed together segmentation, mesh reconstruction, voxelizin’, and brick conversion like a proper ship’s riggin’!
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
