#!/usr/bin/env python3
"""
Visualize point cloud reconstruction results for a Bricked run.

Usage:
    python visualize_pointcloud.py <run_id>
    python visualize_pointcloud.py <run_id> --save output.png
    python visualize_pointcloud.py <run_id> --mongo-uri mongodb://localhost:27017 --db bricked
"""

import argparse
import json
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a Bricked visual-hull point cloud stored in MongoDB"
    )
    parser.add_argument("run_id", help="MongoDB run _id (hex string)")
    parser.add_argument(
        "--mongo-uri", default="mongodb://localhost:27017", help="MongoDB connection URI"
    )
    parser.add_argument("--db", default="bricked", help="MongoDB database name")
    parser.add_argument(
        "--save", metavar="FILE", help="Save the plot to FILE instead of opening a window"
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=20_000,
        help="Sub-sample to at most this many points for rendering (default: 20000)",
    )
    args = parser.parse_args()

    # ── Dependency checks ─────────────────────────────────────────────────────
    try:
        import pymongo
        import gridfs as _gridfs
    except ImportError:
        sys.exit("ERROR: pymongo is not installed.  Run:  pip install pymongo")

    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3-D projection
    except ImportError:
        sys.exit("ERROR: matplotlib is not installed.  Run:  pip install matplotlib")

    from bson import ObjectId
    from bson.errors import InvalidId

    # ── Connect to MongoDB ────────────────────────────────────────────────────
    try:
        client = pymongo.MongoClient(args.mongo_uri, serverSelectionTimeoutMS=5_000)
        client.admin.command("ping")
    except Exception as exc:
        sys.exit(f"ERROR: Could not connect to MongoDB at {args.mongo_uri!r}: {exc}")

    db = client[args.db]
    fs = _gridfs.GridFS(db, collection="images")

    # ── Fetch run ─────────────────────────────────────────────────────────────
    try:
        oid = ObjectId(args.run_id)
    except InvalidId:
        sys.exit(f"ERROR: {args.run_id!r} is not a valid MongoDB ObjectId")

    run = db.runs.find_one({"_id": oid})
    if run is None:
        sys.exit(f"ERROR: No run found with id {args.run_id}")

    print(f"Run status : {run['status']}")

    recon = run.get("reconstruction")
    if recon is None:
        sys.exit(
            "ERROR: This run has no reconstruction data.\n"
            "       Trigger the reconstruct stage first via POST /api/runs/<id>/reconstruct"
        )

    method     = recon.get("method", "unknown")
    grid_size  = recon.get("grid_size", "?")
    point_count= recon.get("point_count", "?")
    print(f"Method     : {method}")
    print(f"Grid size  : {grid_size}³")
    print(f"Points     : {point_count}")

    # ── Load point cloud from GridFS ──────────────────────────────────────────
    try:
        grid_out = fs.get(ObjectId(recon["point_cloud_file_id"]))
        raw      = grid_out.read()
    except Exception as exc:
        sys.exit(f"ERROR: Could not fetch point cloud from GridFS: {exc}")

    data   = json.loads(raw.decode())
    points = np.array([[p["x"], p["y"], p["z"]] for p in data], dtype=np.float32)
    print(f"Loaded     : {len(points)} points")

    # ── Sub-sample if necessary ───────────────────────────────────────────────
    if len(points) > args.max_points:
        idx    = np.random.default_rng(0).choice(len(points), args.max_points, replace=False)
        pts    = points[idx]
        print(f"Displaying : {args.max_points} points (sub-sampled)")
    else:
        pts = points

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 9))

    # ── Main 3-D scatter ──────────────────────────────────────────────────────
    ax3d = fig.add_subplot(221, projection="3d")
    sc   = ax3d.scatter(
        pts[:, 0],
        pts[:, 2],          # use Z as depth axis
        pts[:, 1],          # use Y as vertical axis
        c=pts[:, 1],        # colour by height
        cmap="viridis",
        s=1,
        alpha=0.5,
        linewidths=0,
    )
    fig.colorbar(sc, ax=ax3d, shrink=0.6, label="Y (height)")
    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Z")
    ax3d.set_zlabel("Y")
    ax3d.set_title("3-D View")

    # ── Top-down (XZ) ─────────────────────────────────────────────────────────
    ax_top = fig.add_subplot(222)
    ax_top.scatter(pts[:, 0], pts[:, 2], c=pts[:, 1], cmap="viridis", s=1, alpha=0.4, linewidths=0)
    ax_top.set_xlabel("X")
    ax_top.set_ylabel("Z")
    ax_top.set_title("Top view (XZ)")
    ax_top.set_aspect("equal")

    # ── Front (XY) ────────────────────────────────────────────────────────────
    ax_front = fig.add_subplot(223)
    ax_front.scatter(pts[:, 0], pts[:, 1], c=pts[:, 1], cmap="viridis", s=1, alpha=0.4, linewidths=0)
    ax_front.set_xlabel("X")
    ax_front.set_ylabel("Y")
    ax_front.set_title("Front view (XY)")
    ax_front.set_aspect("equal")

    # ── Side (ZY) ─────────────────────────────────────────────────────────────
    ax_side = fig.add_subplot(224)
    ax_side.scatter(pts[:, 2], pts[:, 1], c=pts[:, 1], cmap="viridis", s=1, alpha=0.4, linewidths=0)
    ax_side.set_xlabel("Z")
    ax_side.set_ylabel("Y")
    ax_side.set_title("Side view (ZY)")
    ax_side.set_aspect("equal")

    run_short = args.run_id[:10]
    fig.suptitle(
        f"Point Cloud — run {run_short}…  |  {len(points):,} pts  |  {method}  |  grid {grid_size}³",
        fontsize=11,
    )
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved to   : {args.save}")
    else:
        plt.show()

    client.close()


if __name__ == "__main__":
    main()
