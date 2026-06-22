"""
Download the selected public broiler/chicken detection datasets from Roboflow Universe.

Roboflow requires a free API key. Get one at https://app.roboflow.com/settings/api
then either:
    set ROBOFLOW_API_KEY=xxxx        (PowerShell: $env:ROBOFLOW_API_KEY="xxxx")
or pass --api-key xxxx

Usage:
    pip install roboflow
    python scripts/download_roboflow_datasets.py --api-key YOUR_KEY
    python scripts/download_roboflow_datasets.py --api-key YOUR_KEY --only broiler_detection

Each dataset is downloaded in YOLOv8 format into data/external/<id>/.

NOTE: Roboflow project slugs/versions can change. If a download fails with "project not
found", open the listed URL in a browser, read the workspace/project/version from the URL,
and update the DATASETS table below. The URL pattern is:
    https://universe.roboflow.com/<workspace>/<project>/dataset/<version>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "external"

# Curated candidates found 2026-06-27. workspace/project/version are best-effort guesses
# from search results; verify in browser if a download fails.
DATASETS = [
    {
        "id": "broiler_detection_innodatatics",
        "url": "https://universe.roboflow.com/innodatatics/broiler-chicken-detection",
        "workspace": "innodatatics",
        "project": "broiler-chicken-detection",
        "version": 1,
        "note": "Broiler-specific, ~179 images",
    },
    {
        "id": "broiler_healthy_sick",
        "url": "https://universe.roboflow.com/technicalresearch/broiler-chicken-healthy-and-sick",
        "workspace": "technicalresearch",
        "project": "broiler-chicken-healthy-and-sick",
        "version": 1,
        "note": "2 classes healthy/sick, ~209 images",
    },
    {
        "id": "chicken_detection_fum",
        "url": "https://universe.roboflow.com/fum-icce/chicken-detection-z6wni",
        "workspace": "fum-icce",
        "project": "chicken-detection-z6wni",
        "version": 5,
        "note": "General chicken, ~157 images",
    },
    {
        "id": "chicken_count",
        "url": "https://universe.roboflow.com/chickendetection-sct5j/chicken-count",
        "workspace": "chickendetection-sct5j",
        "project": "chicken-count",
        "version": 4,
        "note": "Counting / denser, ~100 images",
    },
    {
        "id": "broiler_instance_seg",
        "url": "https://universe.roboflow.com/broiler-data/broiler-ozg7f",
        "workspace": "broiler-data",
        "project": "broiler-ozg7f",
        "version": 1,
        "note": "Instance segmentation -> needs seg2bbox conversion",
    },
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("ROBOFLOW_API_KEY", ""))
    ap.add_argument("--only", default="", help="download only this dataset id")
    ap.add_argument("--format", default="yolov8", help="export format (yolov8, coco, ...)")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: no API key. Set ROBOFLOW_API_KEY or pass --api-key.", file=sys.stderr)
        print("Get a free key at https://app.roboflow.com/settings/api", file=sys.stderr)
        print("\nCandidate datasets (open in browser to verify workspace/project/version):")
        for d in DATASETS:
            print(f"  {d['id']:30s} {d['url']}  ({d['note']})")
        raise SystemExit(2)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow not installed. Run: pip install roboflow", file=sys.stderr)
        raise SystemExit(2)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=args.api_key)

    targets = [d for d in DATASETS if not args.only or d["id"] == args.only]
    if not targets:
        print(f"No dataset matches --only {args.only}. Available: {[d['id'] for d in DATASETS]}")
        raise SystemExit(2)

    for d in targets:
        dest = OUT_DIR / d["id"]
        print(f"\n=== {d['id']} ===\n{d['url']}\n{d['note']}")
        try:
            project = rf.workspace(d["workspace"]).project(d["project"])
            project.version(d["version"]).download(args.format, location=str(dest))
            print(f"OK -> {dest}")
        except Exception as e:
            print(f"FAILED: {e}")
            print(f"  Open {d['url']} in a browser, read workspace/project/version from the")
            print(f"  URL, and update DATASETS in this script, then retry with --only {d['id']}.")


if __name__ == "__main__":
    main()
