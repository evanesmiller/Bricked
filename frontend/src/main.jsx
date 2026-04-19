import React, { useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const MIN_IMAGES = 4;
const MAX_IMAGES = 8;

const meshStages = [
  {
    title: "Segmented Views",
    status: "Waiting for upload",
    copy: "Object masks from each image will surface here.",
  },
  {
    title: "Point Cloud",
    status: "Queued",
    copy: "Structure-from-motion points will anchor the rough model.",
  },
  {
    title: "Mesh Draft",
    status: "Queued",
    copy: "The reconstructed mesh preview will land in this bay.",
  },
];

function formatSize(bytes) {
  if (!bytes) return "0 KB";
  const units = ["B", "KB", "MB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index === 0 ? 0 : 1)} ${units[index]}`;
}

function App() {
  const inputRef = useRef(null);
  const [files, setFiles] = useState([]);
  const [run, setRun] = useState(null);
  const [error, setError] = useState("");
  const [isUploading, setIsUploading] = useState(false);

  const previews = useMemo(
    () =>
      files.map((file) => ({
        file,
        url: URL.createObjectURL(file),
      })),
    [files],
  );

  useEffect(() => {
    return () => previews.forEach((preview) => URL.revokeObjectURL(preview.url));
  }, [previews]);

  const canUpload = files.length >= MIN_IMAGES && files.length <= MAX_IMAGES && !isUploading;

  function updateFiles(nextFiles) {
    const imageFiles = Array.from(nextFiles).filter((file) => file.type.startsWith("image/"));
    setRun(null);
    setError("");

    if (imageFiles.length > MAX_IMAGES) {
      setError(`Choose ${MIN_IMAGES}-${MAX_IMAGES} images for this pass.`);
      setFiles(imageFiles.slice(0, MAX_IMAGES));
      return;
    }

    setFiles(imageFiles);
    if (imageFiles.length > 0 && imageFiles.length < MIN_IMAGES) {
      setError(`Add ${MIN_IMAGES - imageFiles.length} more image${MIN_IMAGES - imageFiles.length === 1 ? "" : "s"} to start.`);
    }
  }

  async function uploadImages() {
    if (!canUpload) return;

    const formData = new FormData();
    files.forEach((file) => formData.append("images", file));

    setIsUploading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/uploads/runs`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(data?.detail ?? "Upload failed. Check the API server and try again.");
      }

      setRun(data);
    } catch (uploadError) {
      setError(uploadError.message);
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
              onUpload={uploadImages}
              previews={previews}
              run={run}
            />
            <PipelinePanel run={run} />
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-6 px-5 pb-10 sm:px-8 lg:grid-cols-[1fr_1fr_1fr] lg:px-10">
        {meshStages.map((stage) => (
          <ModelBay key={stage.title} {...stage} />
        ))}
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

function UploadPanel({ canUpload, error, files, inputRef, isUploading, onFiles, onUpload, previews, run }) {
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
        accept="image/jpeg,image/png,image/webp"
        multiple
        className="hidden"
        onChange={(event) => onFiles(event.target.files)}
      />

      {error && (
        <div className="mt-4 rounded-md border border-rose-200/40 bg-rose-950/50 px-4 py-3 text-sm text-rose-100">
          {error}
        </div>
      )}

      {previews.length > 0 && (
        <div className="mt-5 grid grid-cols-2 gap-3 sm:grid-cols-4">
          {previews.map(({ file, url }) => (
            <figure key={`${file.name}-${file.lastModified}`} className="rounded-md border border-cyan-100/15 bg-[#031318] p-2">
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
          {isUploading ? "Uploading..." : "Create Processing Run"}
        </button>
        <p className="text-sm text-slate-300">
          {canUpload ? "Ready to store images." : `Select ${MIN_IMAGES}-${MAX_IMAGES} images to continue.`}
        </p>
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

function PipelinePanel({ run }) {
  const steps = ["Upload", "Segment", "Reconstruct", "Voxelize", "LEGO Convert"];

  return (
    <aside className="rounded-md border border-teal-200/15 bg-[#04181f]/85 p-5 shadow-abyss backdrop-blur">
      <p className="text-sm font-semibold uppercase tracking-[0.18em] text-teal-200">Processing Current</p>
      <h2 className="mt-2 text-2xl font-bold text-white">Pipeline placeholders are surfaced early.</h2>
      <div className="mt-6 space-y-3">
        {steps.map((step, index) => {
          const isUploaded = Boolean(run) && index === 0;
          return (
            <div key={step} className="flex items-center gap-3 rounded-md border border-cyan-100/10 bg-[#061f27] p-3">
              <div
                className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-md text-sm font-black ${
                  isUploaded ? "bg-emerald-300 text-emerald-950" : "bg-cyan-950 text-cyan-100"
                }`}
              >
                {index + 1}
              </div>
              <div>
                <p className="font-semibold text-white">{step}</p>
                <p className="text-sm text-slate-300">{isUploaded ? "Images stored" : "Coming soon"}</p>
              </div>
            </div>
          );
        })}
      </div>
    </aside>
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

createRoot(document.getElementById("root")).render(<App />);
