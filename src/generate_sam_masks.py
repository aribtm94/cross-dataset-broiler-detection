"""
generate_sam_masks.py — Polygon mask generator dari bbox YOLO via bbox-prompted MobileSAM.

Tujuan (skripsi): menghasilkan mask segmentasi instance (poligon YOLO-seg) dari
bounding box yang SUDAH ADA, dengan memakai MobileSAM (ultralytics) yang di-prompt
oleh tiap bbox. Mask ini dipakai untuk "Lever B": rewarp bbox berbasis MASK saat
rektifikasi MOWA (`src/mowa_rectify.py --seg-labels`), supaya box hasil warp
mengikuti kontur ayam dan TIDAK melebar seperti pada rasterisasi persegi (rectangle)
yang membungkus seluruh box lalu di-warp.

Kontrak korektnes (WAJIB): tepat SATU baris poligon per bbox input, dengan URUTAN
yang SAMA persis dengan file label bbox. Box yang gagal di-SAM diberi fallback
persegi (4 sudut box). Downstream menyelaraskan mask ke box BERDASARKAN INDEKS,
jadi jumlah poligon per gambar HARUS sama dengan jumlah box dan urutan dijaga.

Format output (YOLO-seg), satu baris per box, ternormalisasi 0-1:
  <cls> x1 y1 x2 y2 ... xk yk

Contoh pemakaian (dari root proyek, pakai venv YOLO yang ada ultralytics 8.4.84):
  .venv-yolo/Scripts/python.exe src/generate_sam_masks.py \
      --datasets pio_val \
      --limit 20 \
      --sam-model mobile_sam.pt \
      --device 0

Semua dataset sekaligus (default; chunk 64 box/panggilan — WAJIB, MobileSAM
ultralytics tak andal bila semua box dikirim sekaligus, mask bisa kosong):
  .venv-yolo/Scripts/python.exe src/generate_sam_masks.py

Struktur output:
  data/masks_seg/<dataset_id>/labels/<stem>.txt   poligon YOLO-seg (1 baris / box)
  data/masks_seg/<dataset_id>/_sam_manifest.json   ringkasan per-dataset
  data/masks_seg/_summary.json                      ringkasan semua dataset
(Semuanya di-gitignore via /data/.)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from mowa_rectify import read_yolo_labels, list_images, IMAGE_EXTS  # noqa: E402


# ---------------------------------------------------------------------------
# Definisi dataset — mengikuti layout YOLO yang dipakai di seluruh repo.
# Konvensi: lbl_dirs paralel dengan img_dirs (ganti '/images' -> '/labels').
# ---------------------------------------------------------------------------
def _build_datasets() -> List[Dict]:
    ext = ROOT / "data" / "external"
    fum = ext / "chicken_detection_fum"
    fum_splits = ("test", "valid", "train")
    return [
        {
            "id": "pio_train",
            "img_dirs": [ROOT / "data" / "images" / "train"],
            "lbl_dirs": [ROOT / "data" / "labels" / "train"],
        },
        {
            "id": "pio_val",
            "img_dirs": [ROOT / "data" / "images" / "val"],
            "lbl_dirs": [ROOT / "data" / "labels" / "val"],
        },
        {
            "id": "broiler_instance_seg",
            "img_dirs": [ext / "broiler_instance_seg" / "train" / "images"],
            "lbl_dirs": [ext / "broiler_instance_seg" / "train" / "labels"],
        },
        {
            "id": "chicken_detection_fum",
            "img_dirs": [fum / s / "images" for s in fum_splits],
            "lbl_dirs": [fum / s / "labels" for s in fum_splits],
        },
    ]


def _label_path_for(img_path: Path, lbl_dir: Path) -> Path:
    """File label YOLO untuk sebuah gambar: <lbl_dir>/<stem>.txt."""
    return lbl_dir / (img_path.stem + ".txt")


# ---------------------------------------------------------------------------
# Poligon per-box.
# ---------------------------------------------------------------------------
def _rect_polygon_norm(box_xyxy: np.ndarray, img_w: int, img_h: int) -> List[float]:
    """Fallback: 4 sudut box sebagai poligon, ternormalisasi 0-1.

    Urutan: (x1,y1),(x2,y1),(x2,y2),(x1,y2).
    """
    x1, y1, x2, y2 = (float(box_xyxy[0]), float(box_xyxy[1]),
                      float(box_xyxy[2]), float(box_xyxy[3]))
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    out: List[float] = []
    for x, y in pts:
        out.append(x / img_w)
        out.append(y / img_h)
    return out


def _sam_polygon_norm(poly_px: Optional[np.ndarray], img_w: int, img_h: int) -> Optional[List[float]]:
    """Ubah poligon SAM (PIKSEL Nx2) menjadi list flat ternormalisasi 0-1.

    - Sederhanakan via approxPolyDP ke <=64 titik bila terlalu banyak.
    - Butuh >=3 titik valid (sebelum & sesudah simplifikasi), else None (fallback).
    """
    if poly_px is None:
        return None
    px = np.asarray(poly_px, dtype=np.float32)
    if px.ndim != 2 or px.shape[0] < 3 or px.shape[1] != 2:
        return None
    # Sederhanakan bila titik banyak.
    if px.shape[0] > 64:
        contour = px.reshape(-1, 1, 2).astype(np.float32)
        eps = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if approx.shape[0] >= 3:
            px = approx.astype(np.float32)
        # kalau simplifikasi malah <3 titik, pakai px asli (masih >=3).
    if px.shape[0] < 3:
        return None
    out: List[float] = []
    for x, y in px:
        out.append(float(x) / img_w)
        out.append(float(y) / img_h)
    return out


def _contour_from_mask(mask_bool: np.ndarray) -> Optional[np.ndarray]:
    """Ambil kontur poligon (Nx2 piksel) dari mask biner via findContours (kontur
    terbesar). Return None bila tak ada kontur >=3 titik. Cadangan bila masks.xyn
    kosong padahal masks.data berisi (terjadi pada sebagian versi ultralytics)."""
    m = mask_bool.astype(np.uint8)
    if m.sum() == 0:
        return None
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(np.float32)
    return c if c.shape[0] >= 3 else None


def _run_sam_on_boxes(sam, img_src, boxes_xyxy: np.ndarray, max_boxes_per_call: int,
                      device: str) -> Tuple[List[Optional[np.ndarray]], bool]:
    """Jalankan SAM untuk semua box sebuah gambar, kembalikan poligon PIKSEL (Nx2 atau
    None) per box, URUTAN sama dengan boxes_xyxy.

    Return (polys_px, any_error). Bila SAM error / count mismatch untuk sebuah chunk,
    seluruh box chunk itu diberi None (dipaksa fallback oleh pemanggil).

    CATATAN penting (diverifikasi 2026-07-09): MobileSAM ultralytics 8.4.84
    TIDAK ANDAL bila SEMUA box dikirim dalam satu panggilan besar (kadang seluruh
    mask kosong). Chunk <=64 stabil 100%. Karena itu default max_boxes_per_call=64.
    Param `half`/`quantize` di-DROP: usang & memicu mask kosong. Bila masks.xyn[k]
    kosong padahal masks.data[k] berisi, kontur diambil manual dari masks.data.
    """
    n = int(boxes_xyxy.shape[0])
    polys: List[Optional[np.ndarray]] = [None] * n
    any_error = False

    if n == 0:
        return polys, any_error

    if max_boxes_per_call and max_boxes_per_call > 0:
        chunks = [(i, min(i + max_boxes_per_call, n))
                  for i in range(0, n, max_boxes_per_call)]
    else:
        chunks = [(0, n)]

    for (start, end) in chunks:
        chunk_boxes = boxes_xyxy[start:end]
        expected = end - start
        try:
            res = sam(img_src, bboxes=chunk_boxes.tolist(), verbose=False, device=device)
            masks = res[0].masks if res else None
            if masks is None:
                any_error = True
                continue
            xyn = masks.xyn
            data = masks.data  # (K,H,W) bool/float, koordinat PIKSEL untuk fallback kontur
            if xyn is None or len(xyn) != expected:
                # Alignment tak terpercaya -> fallback seluruh chunk.
                any_error = True
                continue
            for k in range(expected):
                pk = np.asarray(xyn[k], dtype=np.float32)
                if pk.ndim == 2 and pk.shape[0] >= 3:
                    # xyn ternormalisasi 0-1 -> ke piksel di pemanggil? tidak: simpan
                    # sebagai piksel di sini agar seragam dgn cadangan kontur.
                    ph, pw = data.shape[-2], data.shape[-1]
                    px = pk.copy()
                    px[:, 0] *= pw
                    px[:, 1] *= ph
                    polys[start + k] = px
                elif k < data.shape[0]:
                    polys[start + k] = _contour_from_mask(data[k].cpu().numpy())
                # else tetap None -> fallback persegi oleh pemanggil
        except Exception as exc:  # noqa: BLE001
            any_error = True
            print(f"    [warn] SAM gagal untuk chunk box [{start}:{end}]: {exc}")
            continue

    return polys, any_error


def _write_seg_labels(path: Path, classes: List[int], lines_coords: List[List[float]]) -> None:
    """Tulis file YOLO-seg: satu baris '<cls> x1 y1 ...' per box (ternormalisasi)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for cls, coords in zip(classes, lines_coords):
        vals = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{int(cls)} {vals}")
    path.write_text(("\n".join(lines) + ("\n" if lines else "")), encoding="utf-8")


# ---------------------------------------------------------------------------
# Proses satu dataset.
# ---------------------------------------------------------------------------
def process_dataset(sam, ds: Dict, out_root: Path, limit: int,
                    max_boxes_per_call: int, device: str) -> Dict:
    ds_id = ds["id"]
    out_lbl_dir = out_root / ds_id / "labels"
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes = 0
    n_fallback = 0
    t0 = time.time()

    print(f"\n=== Dataset '{ds_id}' ===")
    for img_dir, lbl_dir in zip(ds["img_dirs"], ds["lbl_dirs"]):
        if not img_dir.exists():
            print(f"  [skip] folder gambar tidak ada: {img_dir}")
            continue
        images = list_images(img_dir)
        for img_path in images:
            if limit and limit > 0 and n_images >= limit:
                break
            n_images += 1

            out_path = _label_path_for(img_path, out_lbl_dir)
            lbl_path = _label_path_for(img_path, lbl_dir)

            # Baca gambar untuk dimensi (dan validasi keterbacaan).
            img = cv2.imread(str(img_path))
            if img is None:
                # Gambar tak terbaca: kita tak tahu ukuran & tak bisa SAM.
                # Tulis file kosong (tak ada dimensi untuk fallback) dan lanjut.
                print(f"  [warn] gambar tak terbaca, tulis label kosong: {img_path.name}")
                _write_seg_labels(out_path, [], [])
                continue
            img_h, img_w = img.shape[0], img.shape[1]

            classes, boxes_xyxy = read_yolo_labels(lbl_path, img_w, img_h)
            n = int(boxes_xyxy.shape[0])
            if n == 0:
                _write_seg_labels(out_path, [], [])
                if n_images % 20 == 0:
                    print(f"  [{n_images}] {img_path.name}: 0 box "
                          f"(fallback={n_fallback})")
                continue
            n_boxes += n

            polys_px, _err = _run_sam_on_boxes(
                sam, str(img_path), boxes_xyxy, max_boxes_per_call, device)

            lines_coords: List[List[float]] = []
            for i in range(n):
                coords = _sam_polygon_norm(polys_px[i], img_w, img_h)
                if coords is None:
                    coords = _rect_polygon_norm(boxes_xyxy[i], img_w, img_h)
                    n_fallback += 1
                lines_coords.append(coords)

            _write_seg_labels(out_path, classes, lines_coords)

            if n_images % 20 == 0:
                print(f"  [{n_images}] {img_path.name}: {n} box "
                      f"(fallback={n_fallback})")

        if limit and limit > 0 and n_images >= limit:
            break

    seconds = round(time.time() - t0, 2)
    fallback_rate = round(n_fallback / n_boxes, 6) if n_boxes else 0.0
    manifest = {
        "id": ds_id,
        "sam_model": getattr(sam, "_sam_model_name", None),
        "n_images": n_images,
        "n_boxes": n_boxes,
        "n_fallback": n_fallback,
        "fallback_rate": fallback_rate,
        "seconds": seconds,
    }
    manifest_path = out_root / ds_id / "_sam_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"--- '{ds_id}': {n_images} gambar, {n_boxes} box, "
          f"{n_fallback} fallback ({fallback_rate:.3%}), {seconds}s ---")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate poligon mask YOLO-seg dari bbox YOLO via bbox-prompted "
                    "MobileSAM (ultralytics). Satu poligon per box, urutan dijaga.")
    parser.add_argument("--datasets", type=str, default="",
                        help="Subset id dataset (dipisah koma). Default: semua.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Batas jumlah gambar per dataset (0 = semua).")
    parser.add_argument("--sam-model", type=str, default="mobile_sam.pt",
                        help="Bobot SAM ultralytics (auto-download saat pertama).")
    parser.add_argument("--device", type=str, default="0",
                        help="Device ultralytics (mis. '0' untuk GPU, 'cpu').")
    parser.add_argument("--max-boxes-per-call", type=int, default=64,
                        help="Chunk box per panggilan SAM (default 64). MobileSAM "
                             "ultralytics TIDAK ANDAL bila semua box dikirim sekaligus "
                             "(mask bisa kosong); <=64 stabil. 0 = semua sekaligus (tak disarankan).")
    args = parser.parse_args()

    all_datasets = _build_datasets()
    if args.datasets.strip():
        wanted = {s.strip() for s in args.datasets.split(",") if s.strip()}
        datasets = [d for d in all_datasets if d["id"] in wanted]
        missing = wanted - {d["id"] for d in datasets}
        if missing:
            print(f"[warn] id dataset tak dikenal diabaikan: {sorted(missing)}")
        if not datasets:
            print(f"[error] tidak ada dataset valid dari --datasets={args.datasets}")
            return 2
    else:
        datasets = all_datasets

    from ultralytics import SAM

    sam = SAM(args.sam_model)
    # Simpan nama model untuk manifest (ultralytics tidak menyediakan API stabil).
    try:
        sam._sam_model_name = args.sam_model
    except Exception:  # noqa: BLE001
        pass

    out_root = ROOT / "data" / "masks_seg"
    out_root.mkdir(parents=True, exist_ok=True)

    manifests: List[Dict] = []
    for ds in datasets:
        m = process_dataset(sam, ds, out_root, args.limit,
                            args.max_boxes_per_call, args.device)
        manifests.append(m)

    summary = {
        "sam_model": args.sam_model,
        "device": args.device,
        "max_boxes_per_call": args.max_boxes_per_call,
        "limit": args.limit,
        "datasets": manifests,
        "n_images_total": sum(m["n_images"] for m in manifests),
        "n_boxes_total": sum(m["n_boxes"] for m in manifests),
        "n_fallback_total": sum(m["n_fallback"] for m in manifests),
    }
    total_boxes = summary["n_boxes_total"]
    summary["fallback_rate_total"] = (
        round(summary["n_fallback_total"] / total_boxes, 6) if total_boxes else 0.0)
    (out_root / "_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== SELESAI: {len(manifests)} dataset, "
          f"{summary['n_images_total']} gambar, {summary['n_boxes_total']} box, "
          f"{summary['n_fallback_total']} fallback "
          f"({summary['fallback_rate_total']:.3%}) ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
