"""
Prepare the NESTLER Poultry Behaviour dataset for MASSA AYAM generalizability testing.

Input:
    data/external/nestler_poultry_behaviour.zip

Output:
    data/external/nestler_yolo/
      images/val/*.jpg
      labels/val/*.txt
      dataset.yaml
      metadata.json

What it does:
- Reads NESTLER zip without fully extracting all videos.
- Parses each annotations_*.json file.
- Extracts sampled frames from the corresponding job_*.mp4 / job_*_400frames.mp4.
- Converts bbox annotations to YOLO bbox labels.

Important:
The exact NESTLER annotation JSON schema may differ. This script includes schema
detection for common COCO/CVAT-like structures, and prints the detected schema. If it
cannot parse, inspect the reported keys and extend extract_annotations().
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
ZIP_DEFAULT = ROOT / "data" / "external" / "nestler_poultry_behaviour.zip"
OUT_DEFAULT = ROOT / "data" / "external" / "nestler_yolo"


BBox = Tuple[int, int, float, float, float, float]  # frame, class_id, x, y, w, h absolute pixels


def find_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def print_json_shape(obj: Any, prefix: str = "root", depth: int = 0) -> None:
    if depth > 2:
        return
    if isinstance(obj, dict):
        keys = list(obj.keys())[:15]
        print(f"{prefix}: dict keys={keys}")
        for k in keys[:5]:
            print_json_shape(obj[k], f"{prefix}.{k}", depth + 1)
    elif isinstance(obj, list):
        print(f"{prefix}: list len={len(obj)}")
        if obj:
            print_json_shape(obj[0], f"{prefix}[0]", depth + 1)
    else:
        print(f"{prefix}: {type(obj).__name__}={str(obj)[:80]}")


def class_id_from_label(label: str, class_map: Dict[str, int]) -> int:
    label = str(label).strip() or "chicken"
    if label not in class_map:
        class_map[label] = len(class_map)
    return class_map[label]


def normalize_bbox_xywh(bbox: Iterable[float]) -> Tuple[float, float, float, float]:
    vals = list(map(float, bbox))
    if len(vals) != 4:
        raise ValueError("bbox must have 4 values")
    x, y, w, h = vals
    # If bbox looks like xyxy (x2 > x and y2 > y but width too large not known), caller can adapt.
    return x, y, w, h


def extract_annotations(data: Any) -> Tuple[Dict[int, List[Tuple[int, float, float, float, float]]], Dict[str, int], Dict[str, Any]]:
    """Return frame -> [(class_id, x,y,w,h absolute)] plus class map.

    Supports common shapes:
    1) NESTLER native: {frames:[{frame_index, tracks_bbox:[[x1,y1,x2,y2,track_id,assembly_id]], actions:{...}}]}
    2) COCO-like: {images:[{id,file_name,width,height}], annotations:[{image_id,bbox,category_id}], categories:[]}
    3) CVAT-like list/dict with objects containing frame + bbox/points/label
    4) Fallback: recursively find dicts with both frame-ish and bbox-ish fields.
    """
    by_frame: Dict[int, List[Tuple[int, float, float, float, float]]] = defaultdict(list)
    class_map: Dict[str, int] = {}
    meta: Dict[str, Any] = {"schema": "unknown"}

    # NESTLER native schema. tracks_bbox rows are [x1, y1, x2, y2, track_id, assembly_id].
    # For this project, we collapse behaviour labels into one "chicken" class because the
    # downstream weight/anomaly pipeline needs body-size boxes, not action classes.
    if isinstance(data, dict) and isinstance(data.get("frames"), list):
        first_frame = next((f for f in data["frames"] if isinstance(f, dict)), None)
        if first_frame and "tracks_bbox" in first_frame:
            meta["schema"] = "nestler_native_tracks_bbox"
            meta["frame_width"] = data.get("frame_width")
            meta["frame_height"] = data.get("frame_height")
            class_map["chicken"] = 0
            action_counts: Dict[str, int] = defaultdict(int)
            for frame_obj in data.get("frames", []):
                if not isinstance(frame_obj, dict):
                    continue
                frame_idx = int(frame_obj.get("frame_index", 0))
                actions = frame_obj.get("actions") or {}
                if isinstance(actions, dict):
                    for action in actions.values():
                        action_counts[str(action)] += 1
                for row in frame_obj.get("tracks_bbox") or []:
                    if not isinstance(row, list) or len(row) < 4:
                        continue
                    x1, y1, x2, y2 = map(float, row[:4])
                    w = x2 - x1
                    h = y2 - y1
                    if w <= 0 or h <= 0:
                        continue
                    by_frame[frame_idx].append((0, x1, y1, w, h))
            meta["action_counts"] = dict(sorted(action_counts.items()))
            return by_frame, class_map, meta

    # COCO-like image-level annotations. Frame number derived from image file stem if possible.
    if isinstance(data, dict) and "annotations" in data and isinstance(data.get("annotations"), list):
        meta["schema"] = "coco_like"
        images_by_id = {im.get("id"): im for im in data.get("images", []) if isinstance(im, dict)}
        cat_by_id = {c.get("id"): c.get("name", str(c.get("id"))) for c in data.get("categories", []) if isinstance(c, dict)}
        for ann in data.get("annotations", []):
            if not isinstance(ann, dict) or "bbox" not in ann:
                continue
            im = images_by_id.get(ann.get("image_id"), {})
            fname = str(im.get("file_name", ann.get("image_id", "0")))
            digits = "".join(ch for ch in Path(fname).stem if ch.isdigit())
            frame = int(digits[-6:] or digits or 0)
            label = cat_by_id.get(ann.get("category_id"), str(ann.get("category_id", "chicken")))
            cls = class_id_from_label(label, class_map)
            x, y, w, h = normalize_bbox_xywh(ann["bbox"])
            by_frame[frame].append((cls, x, y, w, h))
        return by_frame, class_map, meta

    # Recursive fallback: look for dicts with frame + bbox-ish fields.
    candidates = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            lower = {str(k).lower(): k for k in o.keys()}
            has_frame = any(k in lower for k in ["frame", "frame_id", "image_id", "frame_number", "timestamp"])
            has_bbox = any(k in lower for k in ["bbox", "box", "bounding_box", "rectangle"])
            has_points = any(k in lower for k in ["points", "polygon"])
            if has_frame and (has_bbox or has_points):
                candidates.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    meta["schema"] = "recursive_bbox_fallback"
    meta["candidate_objects"] = len(candidates)

    for obj in candidates:
        lower = {str(k).lower(): k for k in obj.keys()}
        frame_key = next((lower[k] for k in ["frame", "frame_id", "image_id", "frame_number", "timestamp"] if k in lower), None)
        frame_raw = obj.get(frame_key, 0)
        try:
            frame = int(float(frame_raw))
        except Exception:
            digits = "".join(ch for ch in str(frame_raw) if ch.isdigit())
            frame = int(digits[-6:] or digits or 0)

        label_key = next((lower[k] for k in ["label", "category", "class", "name"] if k in lower), None)
        cls = class_id_from_label(str(obj.get(label_key, "chicken")), class_map)

        bbox_key = next((lower[k] for k in ["bbox", "box", "bounding_box", "rectangle"] if k in lower), None)
        if bbox_key:
            b = obj[bbox_key]
            if isinstance(b, dict):
                if all(k in b for k in ["x", "y", "width", "height"]):
                    x, y, w, h = float(b["x"]), float(b["y"]), float(b["width"]), float(b["height"])
                elif all(k in b for k in ["left", "top", "width", "height"]):
                    x, y, w, h = float(b["left"]), float(b["top"]), float(b["width"]), float(b["height"])
                elif all(k in b for k in ["x1", "y1", "x2", "y2"]):
                    x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                    x, y, w, h = x1, y1, x2 - x1, y2 - y1
                else:
                    continue
            else:
                x, y, w, h = normalize_bbox_xywh(b)
            by_frame[frame].append((cls, x, y, w, h))
            continue

        points_key = next((lower[k] for k in ["points", "polygon"] if k in lower), None)
        pts = obj.get(points_key)
        if pts:
            if isinstance(pts, list) and pts and isinstance(pts[0], dict):
                xs = [float(p.get("x", p.get("X", 0))) for p in pts]
                ys = [float(p.get("y", p.get("Y", 0))) for p in pts]
            else:
                vals = [float(v) for v in pts]
                xs = vals[0::2]
                ys = vals[1::2]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            by_frame[frame].append((cls, x1, y1, x2 - x1, y2 - y1))

    return by_frame, class_map, meta


def cv2_available() -> bool:
    try:
        import cv2  # noqa: F401
        return True
    except Exception:
        return False


def video_size(video: Path) -> Tuple[int, int]:
    if shutil.which("ffprobe"):
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(video)]
        out = subprocess.check_output(cmd, text=True).strip()
        w, h = out.split("x")
        return int(w), int(h)

    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return w, h


def extract_frame(video: Path, frame_idx: int, out_img: Path) -> None:
    out_img.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg"):
        # Select frame by index. This is slower but robust enough for sampled conversion.
        vf = f"select=eq(n\\,{frame_idx})"
        cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf", vf, "-vsync", "0", "-frames:v", "1", str(out_img), "-y"]
        subprocess.run(cmd, check=True)
        return

    import cv2
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot extract frame {frame_idx} from {video}")
    if not cv2.imwrite(str(out_img), frame):
        raise RuntimeError(f"Cannot write frame image: {out_img}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=str(ZIP_DEFAULT))
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--max-frames-per-video", type=int, default=80,
                    help="sample up to N annotated frames per video")
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    out = Path(args.out)
    if not zip_path.exists():
        raise SystemExit(f"Missing NESTLER zip: {zip_path}")

    with zipfile.ZipFile(zip_path) as z:
        json_names = [n for n in z.namelist() if n.endswith(".json") and "annotations_" in n]
        print(f"annotation json files: {len(json_names)}")
        if not json_names:
            raise SystemExit("No annotations_*.json found")

        first = json.loads(z.read(json_names[0]).decode("utf-8"))
        print("\n=== First JSON shape ===")
        print_json_shape(first)
        if args.inspect_only:
            return

        if not find_ffmpeg() and not cv2_available():
            raise SystemExit("Neither ffmpeg nor OpenCV is available. Install ffmpeg or opencv-python first.")

        tmp = Path(tempfile.mkdtemp(prefix="nestler_extract_"))
        all_classes: Dict[str, int] = {}
        summary = []
        total_frames = 0
        total_boxes = 0

        try:
            for jname in json_names:
                job_dir = Path(jname).parent.as_posix()
                job_id = Path(job_dir).name.replace("job_", "")
                data = json.loads(z.read(jname).decode("utf-8"))
                by_frame, class_map, meta = extract_annotations(data)
                for cname in class_map:
                    if cname not in all_classes:
                        all_classes[cname] = len(all_classes)

                # Prefer 400-frame video for smaller extraction if present.
                video_candidates = [
                    f"{job_dir}/job_{job_id}_400frames.mp4",
                    f"{job_dir}/job_{job_id}.mp4",
                ]
                vname = next((v for v in video_candidates if v in z.namelist()), None)
                if not vname:
                    print(f"WARN no video for {jname}")
                    continue
                video_tmp = tmp / Path(vname).name
                video_tmp.write_bytes(z.read(vname))
                w_px, h_px = video_size(video_tmp)

                frames = sorted(by_frame.keys())[: args.max_frames_per_video]
                print(f"job {job_id}: schema={meta.get('schema')} annotated_frames={len(by_frame)} sampled={len(frames)} size={w_px}x{h_px}")

                for frame in frames:
                    img_name = f"nestler_{job_id}_f{frame:06d}.jpg"
                    img_out = out / "images" / "val" / img_name
                    lbl_out = out / "labels" / "val" / f"nestler_{job_id}_f{frame:06d}.txt"
                    extract_frame(video_tmp, frame, img_out)
                    lines = []
                    for cls, x, y, bw, bh in by_frame[frame]:
                        # clamp absolute bbox and normalize
                        x = max(0.0, min(float(x), w_px))
                        y = max(0.0, min(float(y), h_px))
                        bw = max(0.0, min(float(bw), w_px - x))
                        bh = max(0.0, min(float(bh), h_px - y))
                        if bw <= 0 or bh <= 0:
                            continue
                        xc = (x + bw / 2) / w_px
                        yc = (y + bh / 2) / h_px
                        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw / w_px:.6f} {bh / h_px:.6f}")
                    lbl_out.parent.mkdir(parents=True, exist_ok=True)
                    lbl_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    total_frames += 1
                    total_boxes += len(lines)

                summary.append({"job": job_id, "json": jname, "video": vname, "schema": meta, "frames_sampled": len(frames)})

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    (out / "dataset.yaml").write_text(
        "path: .\ntrain: images/val\nval: images/val\nnc: 1\nnames: ['chicken']\n",
        encoding="utf-8",
    )
    (out / "metadata.json").write_text(json.dumps({
        "source": "NESTLER - Poultry Behaviour Analytics Detection Dataset",
        "doi": "10.5281/zenodo.20924893",
        "license": "CC-BY-4.0",
        "total_frames_sampled": total_frames,
        "total_boxes": total_boxes,
        "class_map_detected": all_classes,
        "jobs": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nDONE -> {out}")
    print(f"frames={total_frames}, boxes={total_boxes}")


if __name__ == "__main__":
    main()
