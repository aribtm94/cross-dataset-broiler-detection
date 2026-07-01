"""
radial_distort_augment.py — Augmentasi distorsi radial ACAK (barrel/pincushion) pada
train set PIO dengan RE-WARP bbox, + data.yaml fine-tune. Task 3d.

Tujuan (skripsi):
  Hipotesis alternatif terhadap rektifikasi test-time (MOWA): alih-alih meluruskan
  gambar saat inferensi, kita ADAPTASI detektor agar TAHAN-DISTORSI dengan melatihnya
  pada gambar train yang diberi distorsi radial acak (barrel/pincushion) beserta bbox
  yang di-warp konsisten. Ini "arm" adapt-detector (no test-time rectify) dari rencana.
  Dasar: WoodScape (ICCV 2019, Yogamani dkk.) & FisheyeYOLO — detektor yang dilatih
  pada citra terdistorsi belajar fitur invarian-distorsi tanpa perlu meluruskan input.
  Lihat juga arXiv:2507.16254 (survei deteksi objek pada citra fisheye/terdistorsi).

Model distorsi:
  Murni OpenCV/NumPy, TANPA torch/GPU. Model radial polinomial (Brown, orde k1[,k2]):
  koordinat dinormalisasi ke pusat gambar, dibagi setengah-diagonal (radius sudut = 1).
  Peta yang dibangun untuk `cv2.remap` adalah BACKWARD map: untuk tiap pixel OUTPUT pada
  radius r_o, sampel diambil dari radius sumber r_s = r_o * (1 + k1*r_o^2 + k2*r_o^4).
  Konsekuensi tanda (dipakai konvensi tugas):
    k1 > 0  -> barrel     (pusat membesar, tepi termampat; garis melengkung keluar)
    k1 < 0  -> pincushion  (pusat mengecil, tepi meregang)
  k1 diacak per gambar/kopi dalam --k1-range (mis. -0.3..0.3) agar detektor melihat
  ragam distorsi. Border diisi hitam (BORDER_CONSTANT): sudut hitam wajar pada barrel
  dan tidak menimbulkan konten palsu yang bisa salah-terlabel.

RE-WARP bbox:
  Di bawah distorsi radial nonlinier, bbox axis-aligned TIDAK lagi persegi. Metode robust:
  ambil banyak titik SEPANJANG tiap sisi kotak (bukan hanya 4 sudut — sisi bisa melengkung
  keluar dari kotak sudut, terutama dekat tepi frame), petakan MAJU (original -> output)
  dengan mapping yang KONSISTEN terhadap remap (invers 1-D dari fungsi radial backward,
  dicari via Newton karena murni radial sehingga hanya skalar per titik), lalu ambil
  min/max x,y dari titik-titik terpetakan sebagai bbox axis-aligned baru. Kotak diklip ke
  batas frame; kotak yang keluar frame total atau kolaps (<1px) dibuang.
  Pilihan sampel-sisi vs sudut: dipakai --edge-samples titik per sisi (default 8) karena
  distorsi menggeser titik tengah sisi lebih jauh dari sudut; memakai 4 sudut saja akan
  under/over-estimasi kotak. Lihat forward_distort_points() & warp_boxes().

Determinisme:
  Lingkungan melarang Math.random/acak tak-terbibit. Semua keacakan lewat
  random.Random((seed, idx)) yang dibibit PER GAMBAR menurut indeks -> run reproducible.

I/O & CLI:
  --input       data/images/train   (gambar sumber)
  --labels      data/labels/train   (label YOLO, kelas tunggal 0)
  --output      data/augmented/pio_train_radial
  --k1-range    dua float (default -0.3 0.3)
  --k2-range    dua float (default 0 0)
  --copies      jumlah kopi terdistorsi per gambar (default 1)
  --seed        int (WAJIB utk determinisme; default 0)
  --limit       proses N gambar pertama saja (0 = semua)
  --ext         ekstensi gambar output (default .jpg)
  --edge-samples titik sampel per sisi kotak (default 8)

Output:
  <output>/images/<stem>_d{c}.jpg          gambar terdistorsi
  <output>/labels/<stem>_d{c}.txt          bbox YOLO ternormalisasi hasil re-warp
  <output>/_radial_pio.yaml                data.yaml fine-tune (train=augmented, val=asli)
  <output>/radial_augment_manifest.json    ringkasan (params, total, boxes in/out/dropped)

Retrain (DIJALANKAN OLEH KOORDINATOR pada GPU setelah merge):
  data.yaml yang diemit (_radial_pio.yaml) menunjuk train -> set teraugmentasi,
  val -> data/images/val ASLI (poin eksperimen: robustness tanpa rektifikasi test-time).

  finetune_rectified.py mem-build yaml-nya sendiri (path rectified hardcoded) dan belum
  punya argumen --data, jadi koordinator punya dua opsi:

  Opsi A (paling ringkas — panggil Ultralytics langsung, hyperparam sama spt finetune):
    .venv-yolo/Scripts/python.exe -c "from ultralytics import YOLO; \
      YOLO(r'train model/runs_compare/cmp_yolov8m/weights/best.pt').train( \
      data=r'data/augmented/pio_train_radial/_radial_pio.yaml', epochs=40, imgsz=960, \
      batch=4, device='0', project=r'train model/runs_radial', name='ft_radial_yolov8m', \
      exist_ok=True, patience=10, verbose=True)"

  Opsi B (wrapper mengikuti pola finetune_rectified.py — tambah pass-through --data):
    setara perintah konvensi berikut TAPI ditodongkan ke yaml radial di atas:
    .venv-yolo/Scripts/python.exe src/finetune_rectified.py \
        --weights "train model/runs_compare/cmp_yolov8m/weights/best.pt" --epochs 40
    (edit build_yaml / tambahkan argumen --data agar memakai _radial_pio.yaml).

  Bobot hasil lalu dievaluasi ulang oleh src/eval_detection.py (arm "adapt-detector").

CATATAN commit: hanya file .py ini yang di-commit. Gambar/yaml teraugmentasi ada di
data/ (gitignored) — JANGAN commit.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_PX = 1.0  # kotak dengan w/h < ini (setelah klip) dianggap kolaps -> dibuang


# ---------------------------------------------------------------------------
# I/O label YOLO (pola sama dgn mowa_rectify.read_yolo_labels/write_yolo_labels)
# ---------------------------------------------------------------------------
def list_images(input_dir: Path) -> List[Path]:
    return sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_yolo_labels(path: Path, img_w: int, img_h: int) -> Tuple[List[int], np.ndarray]:
    """Baca .txt YOLO -> (classes, boxes_xyxy Nx4 pixel). Baris tak valid dilewati."""
    classes: List[int] = []
    rows: List[Tuple[float, float, float, float]] = []
    if not path.exists():
        return classes, np.zeros((0, 4), dtype=np.float32)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            continue
        x1 = (cx - w / 2.0) * img_w
        y1 = (cy - h / 2.0) * img_h
        x2 = (cx + w / 2.0) * img_w
        y2 = (cy + h / 2.0) * img_h
        classes.append(cls)
        rows.append((x1, y1, x2, y2))
    boxes = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 4), dtype=np.float32)
    return classes, boxes


def write_yolo_labels(path: Path, warped: List[Tuple[int, float, float, float, float]],
                      img_w: int, img_h: int) -> int:
    """Tulis bbox hasil warp (idx, x1,y1,x2,y2 pixel) ke .txt YOLO ternormalisasi."""
    lines: List[str] = []
    for (cls, x1, y1, x2, y2) in warped:
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        if bw <= 0 or bh <= 0:
            continue
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


# ---------------------------------------------------------------------------
# Model distorsi radial
# ---------------------------------------------------------------------------
def _norm_factor(w: int, h: int) -> Tuple[float, float, float]:
    """Kembalikan (cx, cy, norm). norm = setengah-diagonal -> radius sudut = 1.0."""
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    norm = 0.5 * float(np.hypot(w, h))
    return cx, cy, norm


def build_radial_maps(h: int, w: int, k1: float, k2: float) -> Tuple[np.ndarray, np.ndarray]:
    """BACKWARD map utk cv2.remap. Untuk tiap pixel OUTPUT (u,v), sampel sumber di
    r_s = r_o * (1 + k1*r_o^2 + k2*r_o^4). Return (map_x, map_y) float32 HxW."""
    cx, cy, norm = _norm_factor(w, h)
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                         np.arange(w, dtype=np.float32), indexing="ij")
    x = (xs - cx) / norm
    y = (ys - cy) / norm
    r2 = x * x + y * y
    factor = 1.0 + k1 * r2 + k2 * (r2 * r2)
    map_x = (x * factor) * norm + cx
    map_y = (y * factor) * norm + cy
    return map_x.astype(np.float32), map_y.astype(np.float32)


def forward_distort_points(pts_xy: np.ndarray, cx: float, cy: float, norm: float,
                           k1: float, k2: float, iters: int = 12) -> np.ndarray:
    """Peta MAJU (original -> output) titik Nx2, KONSISTEN dgn build_radial_maps.

    Backward map memakai r_s = r_o*(1 + k1 r_o^2 + k2 r_o^4); artinya konten di radius
    sumber R muncul di output pada r_o yang memenuhi r_o*(1+k1 r_o^2+k2 r_o^4) = R.
    Karena murni radial (arah dipertahankan), invers = pencarian skalar 1-D pada radius
    via Newton (monoton utk k1 in [-0.3,0.3], r<=~1). Return Nx2 koordinat output pixel.
    """
    if len(pts_xy) == 0:
        return pts_xy.astype(np.float32)
    dx = (pts_xy[:, 0] - cx) / norm
    dy = (pts_xy[:, 1] - cy) / norm
    R = np.hypot(dx, dy)
    r = R.copy()  # tebakan awal
    for _ in range(iters):
        r2 = r * r
        f = r * (1.0 + k1 * r2 + k2 * (r2 * r2)) - R
        fp = 1.0 + 3.0 * k1 * r2 + 5.0 * k2 * (r2 * r2)
        fp = np.where(np.abs(fp) < 1e-8, 1e-8, fp)
        r = np.clip(r - f / fp, 0.0, None)
    # scale = r_o / R (arah dipertahankan); R~0 -> titik pusat tak bergerak.
    scale = np.where(R > 1e-8, r / np.where(R > 1e-8, R, 1.0), 1.0)
    out_x = dx * scale * norm + cx
    out_y = dy * scale * norm + cy
    return np.stack([out_x, out_y], axis=1).astype(np.float32)


def warp_boxes(boxes_xyxy: np.ndarray, classes: List[int], w: int, h: int,
               k1: float, k2: float, edge_samples: int
               ) -> Tuple[List[Tuple[int, float, float, float, float]], int]:
    """Re-warp tiap bbox: sampel titik sepanjang 4 sisi, peta maju, ambil min/max.

    Return (list (cls, x1,y1,x2,y2) pixel pada gambar output, jumlah dibuang).
    Dibuang jika kotak keluar frame total atau kolaps (<MIN_BOX_PX) setelah klip.
    """
    out: List[Tuple[int, float, float, float, float]] = []
    dropped = 0
    if len(boxes_xyxy) == 0:
        return out, dropped
    cx, cy, norm = _norm_factor(w, h)
    n = max(2, int(edge_samples) + 1)  # titik per sisi termasuk sudut
    t = np.linspace(0.0, 1.0, n, dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        # Titik sepanjang keempat sisi (sudut inklusif; duplikat tak masalah utk min/max).
        top = np.stack([x1 + t * (x2 - x1), np.full_like(t, y1)], axis=1)
        bot = np.stack([x1 + t * (x2 - x1), np.full_like(t, y2)], axis=1)
        left = np.stack([np.full_like(t, x1), y1 + t * (y2 - y1)], axis=1)
        right = np.stack([np.full_like(t, x2), y1 + t * (y2 - y1)], axis=1)
        edge_pts = np.concatenate([top, bot, left, right], axis=0)

        mapped = forward_distort_points(edge_pts, cx, cy, norm, k1, k2)
        nx1, ny1 = float(mapped[:, 0].min()), float(mapped[:, 1].min())
        nx2, ny2 = float(mapped[:, 0].max()), float(mapped[:, 1].max())

        # Buang bila seluruh kotak di luar frame.
        if nx2 <= 0 or ny2 <= 0 or nx1 >= w or ny1 >= h:
            dropped += 1
            continue
        # Klip ke frame.
        cx1 = max(0.0, nx1)
        cy1 = max(0.0, ny1)
        cx2 = min(float(w), nx2)
        cy2 = min(float(h), ny2)
        if (cx2 - cx1) < MIN_BOX_PX or (cy2 - cy1) < MIN_BOX_PX:
            dropped += 1
            continue
        cls = classes[i] if 0 <= i < len(classes) else 0
        out.append((cls, cx1, cy1, cx2, cy2))
    return out, dropped


# ---------------------------------------------------------------------------
# data.yaml fine-tune (mirror src/finetune_rectified.py::build_yaml)
# ---------------------------------------------------------------------------
def build_yaml(out_path: Path, aug_images_dir: Path) -> Path:
    """Emit data.yaml: train=set teraugmentasi, val=data/images/val ASLI (robustness)."""
    cfg = {
        "path": str(ROOT),
        "train": [str(aug_images_dir)],
        "val": [str(ROOT / "data" / "images" / "val")],
        "nc": 1,
        "names": {0: "pollo"},
    }
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Augmentasi distorsi radial acak (barrel/pincushion) + re-warp bbox.")
    ap.add_argument("--input", type=Path, default=ROOT / "data" / "images" / "train",
                    help="Folder gambar sumber (default data/images/train).")
    ap.add_argument("--labels", type=Path, default=ROOT / "data" / "labels" / "train",
                    help="Folder label YOLO (.txt) (default data/labels/train).")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "augmented" / "pio_train_radial",
                    help="Folder output (dibuat: images/, labels/).")
    ap.add_argument("--k1-range", type=float, nargs=2, default=(-0.3, 0.3),
                    metavar=("LO", "HI"), help="Rentang k1 (neg=pincushion, pos=barrel).")
    ap.add_argument("--k2-range", type=float, nargs=2, default=(0.0, 0.0),
                    metavar=("LO", "HI"), help="Rentang k2 (orde tinggi; default 0 0).")
    ap.add_argument("--copies", type=int, default=1,
                    help="Jumlah kopi terdistorsi per gambar sumber (default 1).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Bibit determinisme (WAJIB; random.Random((seed, idx)) per gambar).")
    ap.add_argument("--limit", type=int, default=0, help="Proses N gambar pertama saja (0 = semua).")
    ap.add_argument("--ext", default=".jpg", help="Ekstensi file output gambar (default .jpg).")
    ap.add_argument("--edge-samples", type=int, default=8,
                    help="Titik sampel per sisi kotak utk re-warp (default 8).")
    args = ap.parse_args()

    if not args.input.is_dir():
        print(f"ERROR: --input bukan folder: {args.input}")
        return 2
    if args.copies < 1:
        print("ERROR: --copies harus >= 1")
        return 2

    k1_lo, k1_hi = sorted(args.k1_range)
    k2_lo, k2_hi = sorted(args.k2_range)

    out_img_dir = args.output / "images"
    out_lbl_dir = args.output / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(args.input)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        print(f"ERROR: tidak ada gambar di {args.input}")
        return 2

    total_out = 0
    boxes_in_total = 0
    boxes_out_total = 0
    boxes_dropped_total = 0
    failed = 0
    t0 = time.time()

    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            print(f"  SKIP (tak terbaca): {img_path.name}")
            continue
        h, w = img.shape[:2]
        src_lbl = args.labels / (img_path.stem + ".txt")
        classes, boxes = read_yolo_labels(src_lbl, w, h)

        # Bibit per gambar berdasarkan indeks -> reproducible tanpa Math.random.
        rng = random.Random((args.seed, idx))
        for c in range(args.copies):
            k1 = rng.uniform(k1_lo, k1_hi)
            k2 = rng.uniform(k2_lo, k2_hi)
            map_x, map_y = build_radial_maps(h, w, k1, k2)
            dist = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            warped, dropped = warp_boxes(boxes, classes, w, h, k1, k2, args.edge_samples)
            boxes_in_total += len(classes)
            boxes_dropped_total += dropped

            stem = f"{img_path.stem}_d{c}"
            cv2.imwrite(str(out_img_dir / (stem + args.ext)), dist)
            n_written = write_yolo_labels(out_lbl_dir / (stem + ".txt"), warped, w, h)
            boxes_out_total += n_written
            total_out += 1

        if (idx + 1) % 20 == 0 or (idx + 1) == len(images):
            dt = time.time() - t0
            print(f"  [{idx + 1}/{len(images)}] out={total_out} "
                  f"boxes {boxes_out_total}/{boxes_in_total} dropped={boxes_dropped_total} "
                  f"({dt:.1f}s, {dt / (idx + 1):.3f}s/img)")

    yaml_path = build_yaml(args.output / "_radial_pio.yaml", out_img_dir)

    elapsed = time.time() - t0
    manifest = {
        "hypothesis": "adapt-detector distortion-invariant (WoodScape ICCV2019 / FisheyeYOLO; "
                      "arXiv:2507.16254). No test-time rectify.",
        "input": str(args.input),
        "labels": str(args.labels),
        "output": str(args.output),
        "data_yaml": str(yaml_path),
        "model": "radial polynomial (Brown k1[,k2]), cv2.remap backward map, norm=half-diagonal",
        "sign_convention": "k1>0 barrel, k1<0 pincushion",
        "k1_range": [k1_lo, k1_hi],
        "k2_range": [k2_lo, k2_hi],
        "copies": args.copies,
        "seed": args.seed,
        "edge_samples": args.edge_samples,
        "border": "BORDER_CONSTANT (0,0,0)",
        "src_images": len(images),
        "total": total_out,
        "failed_reads": failed,
        "boxes_in": boxes_in_total,
        "boxes_out": boxes_out_total,
        "boxes_dropped": boxes_dropped_total,
        "seconds": round(elapsed, 2),
        "sec_per_img": round(elapsed / max(1, len(images)), 4),
        "caveat": (
            "bbox di-warp via peta maju radial (invers Newton dari backward map, sampel per-sisi) "
            "sehingga selaras dgn gambar terdistorsi; kotak keluar-frame/kolaps dibuang. "
            "val pada _radial_pio.yaml sengaja memakai data/images/val ASLI (uji robustness "
            "tanpa rektifikasi test-time)."
        ),
    }
    (args.output / "radial_augment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[radial_augment] SELESAI: {total_out} gambar -> {args.output}")
    print(f"[radial_augment] data.yaml -> {yaml_path}")
    print(f"[radial_augment] manifest  -> {args.output / 'radial_augment_manifest.json'}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
