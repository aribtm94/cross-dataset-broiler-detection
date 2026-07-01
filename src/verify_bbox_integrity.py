"""
verify_bbox_integrity.py — Verifikasi integritas GEOMETRIS bounding box setelah
rektifikasi MOWA (Task 1 pembimbing).

Tujuan:
  Pembimbing meminta bukti bahwa bbox pada gambar HASIL rektifikasi MOWA masih
  benar secara geometris: TIDAK melebar / menyusut secara keliru, dan ayam tidak
  ter-crop keluar frame. Idenya sederhana: "gambar kotak hitam padat di lokasi
  bbox, luruskan, lalu lihat apakah bentuk area hitam itu berubah". Script ini
  mengotomasi cek tersebut untuk 3 dataset dan mengeluarkan metrik kuantitatif.

Metode (didokumentasikan supaya jujur soal apa yang diukur):
  Untuk tiap gambar sampel:
    1. Baca GT bbox (xyxy pixel) via `read_yolo_labels`.
    2. Hitung field warp MOWA SEKALI via `compute_flows` (tps2flow + flow residual
       resolusi asli). Semua warp berikut memakai field yang SAMA agar konsisten
       dengan gambar rectified sebenarnya.
    3. Gambar (bilinear) di-warp via `_apply_full_warp` -> citra rectified (dipakai
       untuk overlay + threshold).
    4. "flow-warp box" = geometri bbox yang ditulis MOWA ke label, via
       `warp_boxes_via_flow` (instance-mask id + NEAREST -> AABB pixel yang tersisa).
       Ini adalah ground-truth-of-transform (persis yang dipakai mowa_rectify.py
       label-mode=warp).
    5. "recovered box" = ukur ulang area kotak secara independen: lukis tiap GT box
       sebagai persegi terisi ber-ID unik (i+1) pada kanvas 1-kanal (latar 0), warp
       dengan field yang SAMA (NEAREST), lalu untuk tiap ID pulihkan AABB + JUMLAH
       pixel yang benar-benar terisi. Dari sini didapat:
         - fill_ratio = pixel_terisi / (rec_w*rec_h) -> seberapa "persegi" area yang
           ter-warp (turun jika kotak jadi melengkung/miring).
         - area_ratio = pixel_terisi / area_flowwarp -> turun jika sebagian kotak
           ter-crop keluar frame / tertutup.
    6. Bandingkan recovered vs flow-warp (sanity, harusnya ~1 karena mekanisme sama):
       width_ratio, height_ratio, IoU. Dan bandingkan warped vs ASLI untuk pertanyaan
       inti pembimbing: widen_w = flowwarp_w/orig_w, widen_h = flowwarp_h/orig_h
       (>1 = melebar, <1 = menyusut).
    7. cropped-flag = (AABB menyentuh tepi frame setelah warp) ATAU (area_ratio < tau).
       dropped = kotak hilang total (ter-warp keluar frame).

  Catatan independensi: recovered (kanvas id) dan flow-warp memakai mekanisme
  NEAREST yang sama, jadi width_ratio/height_ratio/IoU praktis ~1 — itu memang
  cek konsistensi bahwa `warp_boxes_via_flow` setia pada area pixel. Sinyal yang
  benar-benar informatif untuk pembimbing adalah widen_w/widen_h (vs asli),
  fill_ratio (melengkung), area_ratio & cropped/dropped (ter-potong).

Output:
  reports/bbox_integrity/<id>_metrics.csv      (baris per-box)
  reports/bbox_integrity/bbox_integrity_summary.json  (agregat per-dataset + overall)
  reports/bbox_integrity/<id>_overlays/*.png   (~5/dataset: asli+GT vs rectified+recovered)

MOWA butuh CUDA (utils_transform hardcode .cuda()). Tanpa GPU -> exit code 3.

Contoh pemakaian (dari root proyek, pakai venv khusus MOWA):
  .venv-mowa/Scripts/python.exe src/verify_bbox_integrity.py \
      --mowa-root vendor/MOWA \
      --checkpoint vendor/MOWA/checkpoint/mowa_pretrained.pth \
      --limit 30 --datasets chicken_detection_fum,broiler_instance_seg,pio_val \
      --tau 0.6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from common import mean, median, percentile, write_csv, write_json
from mowa_rectify import (
    INPUT_SIZE,
    add_mowa_to_path,
    build_net,
    compute_flows,
    _apply_full_warp,
    load_checkpoint,
    read_yolo_labels,
    warp_boxes_via_flow,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "bbox_integrity"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TPS_POINTS = [10, 12, 14, 16]
OVERLAYS_PER_DATASET = 5
EPS = 0.05  # ambang toleransi widen/shrink (5%)

# Definisi dataset (hardcode meniru DATASETS di eval_detection.py). Semua single-class.
DATASETS: List[Dict] = [
    {
        "id": "pio_val",
        "display": "PIO val (in-domain)",
        "image_dirs": [ROOT / "data" / "images" / "val"],
        "label_dirs": [ROOT / "data" / "labels" / "val"],
        "in_domain": True,
    },
    {
        "id": "broiler_instance_seg",
        "display": "Roboflow broiler_instance_seg (external)",
        "image_dirs": [ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "images"],
        "label_dirs": [ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "labels"],
        "in_domain": False,
    },
    {
        "id": "chicken_detection_fum",
        "display": "Roboflow chicken_detection_fum (external, dense)",
        "image_dirs": [
            ROOT / "data" / "external" / "chicken_detection_fum" / "test" / "images",
            ROOT / "data" / "external" / "chicken_detection_fum" / "valid" / "images",
            ROOT / "data" / "external" / "chicken_detection_fum" / "train" / "images",
        ],
        "label_dirs": [
            ROOT / "data" / "external" / "chicken_detection_fum" / "test" / "labels",
            ROOT / "data" / "external" / "chicken_detection_fum" / "valid" / "labels",
            ROOT / "data" / "external" / "chicken_detection_fum" / "train" / "labels",
        ],
        "in_domain": False,
    },
]


def collect_pairs(ds: Dict, limit: int) -> List[Tuple[Path, Path]]:
    """Kumpulkan pasangan (gambar, label) untuk satu dataset, dibatasi `limit`."""
    pairs: List[Tuple[Path, Path]] = []
    for img_dir, lbl_dir in zip(ds["image_dirs"], ds["label_dirs"]):
        if not img_dir.is_dir():
            continue
        imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        for img_path in imgs:
            pairs.append((img_path, lbl_dir / (img_path.stem + ".txt")))
            if limit > 0 and len(pairs) >= limit:
                return pairs
    return pairs


def _prepare_inputs(img_bgr: np.ndarray, device: torch.device):
    """Siapkan tensor input MOWA (meniru rectify_one) -> (input1_t, input2_t, mask_t)."""
    input1 = np.transpose(img_bgr.astype(np.float32) / 255.0, (2, 0, 1))[None]  # 1,3,H,W
    resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    input2 = np.transpose(resized, (2, 0, 1))[None]  # 1,3,256,256
    mask = np.ones((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    input1_t = torch.from_numpy(input1).float().to(device)
    input2_t = torch.from_numpy(input2).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)
    return input1_t, input2_t, mask_t


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """IoU dua AABB (x1,y1,x2,y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def process_image(net, img_bgr: np.ndarray, boxes: np.ndarray, device: torch.device,
                  tps_cap: int, tau: float, use_fp16: bool):
    """Proses satu gambar -> (rectified_bgr, warped_id_np, rows).

    rows: list dict metrik per-box. warped_id_np dipakai untuk overlay.
    """
    ori_h, ori_w = img_bgr.shape[:2]
    input1_t, input2_t, mask_t = _prepare_inputs(img_bgr, device)

    tps2flow, flow = compute_flows(net, input2_t, mask_t, ori_h, ori_w, TPS_POINTS,
                                   device, use_fp16, tps_cap=tps_cap)

    # Citra rectified (bilinear) — sama seperti yang dikonsumsi detektor.
    warp = _apply_full_warp(input1_t, tps2flow, flow, mode="bilinear")[0]
    rect_bgr = (warp.clamp(0, 1) * 255.0).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
    if rect_bgr.shape[:2] != (ori_h, ori_w):
        rect_bgr = cv2.resize(rect_bgr, (ori_w, ori_h))

    # flow-warp box (geometri label MOWA) -> dict idx -> (x1,y1,x2,y2).
    flowwarp_list = warp_boxes_via_flow(boxes, tps2flow, flow, ori_h, ori_w, device)
    flowwarp = {idx: (x1, y1, x2, y2) for (idx, x1, y1, x2, y2) in flowwarp_list}

    # Kanvas id (independen): warp NEAREST, pulihkan AABB + jumlah pixel terisi.
    id_map = np.zeros((ori_h, ori_w), dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        xi1 = max(0, min(ori_w - 1, int(round(x1))))
        yi1 = max(0, min(ori_h - 1, int(round(y1))))
        xi2 = max(0, min(ori_w, int(round(x2))))
        yi2 = max(0, min(ori_h, int(round(y2))))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        id_map[yi1:yi2, xi1:xi2] = float(i + 1)
    id_t = torch.from_numpy(id_map)[None, None].to(device)
    warped_id = _apply_full_warp(id_t, tps2flow, flow, mode="nearest")[0, 0]
    warped_id_np = warped_id.round().cpu().numpy().astype(np.int32)

    rows: List[Dict] = []
    for i, (ox1, oy1, ox2, oy2) in enumerate(boxes):
        orig_w = float(ox2 - ox1)
        orig_h = float(oy2 - oy1)
        row: Dict = {
            "box_idx": i,
            "orig_w": round(orig_w, 2),
            "orig_h": round(orig_h, 2),
        }
        ys, xs = np.where(warped_id_np == (i + 1))
        fw = flowwarp.get(i)
        if xs.size == 0 or fw is None:
            row.update({
                "dropped": 1, "cropped": 1, "touches_border": "",
                "recovered_w": "", "recovered_h": "", "flowwarp_w": "", "flowwarp_h": "",
                "filled_px": int(xs.size), "fill_ratio": "", "area_ratio": "",
                "width_ratio": "", "height_ratio": "", "iou": "",
                "widen_w": "", "widen_h": "",
            })
            rows.append(row)
            continue

        rx1, ry1 = int(xs.min()), int(ys.min())
        rx2, ry2 = int(xs.max()) + 1, int(ys.max()) + 1
        rec_w, rec_h = float(rx2 - rx1), float(ry2 - ry1)
        filled = int(xs.size)
        fill_ratio = filled / (rec_w * rec_h) if rec_w * rec_h > 0 else 0.0

        fw_w = float(fw[2] - fw[0])
        fw_h = float(fw[3] - fw[1])
        fw_area = fw_w * fw_h
        area_ratio = filled / fw_area if fw_area > 0 else 0.0

        touches_border = int(rx1 <= 0 or ry1 <= 0 or rx2 >= ori_w or ry2 >= ori_h)
        cropped = int(bool(touches_border) or area_ratio < tau)

        row.update({
            "dropped": 0,
            "cropped": cropped,
            "touches_border": touches_border,
            "recovered_w": round(rec_w, 2),
            "recovered_h": round(rec_h, 2),
            "flowwarp_w": round(fw_w, 2),
            "flowwarp_h": round(fw_h, 2),
            "filled_px": filled,
            "fill_ratio": round(fill_ratio, 4),
            "area_ratio": round(area_ratio, 4),
            "width_ratio": round(rec_w / fw_w, 4) if fw_w > 0 else "",
            "height_ratio": round(rec_h / fw_h, 4) if fw_h > 0 else "",
            "iou": round(_iou((rx1, ry1, rx2, ry2), fw), 4),
            "widen_w": round(fw_w / orig_w, 4) if orig_w > 0 else "",
            "widen_h": round(fw_h / orig_h, 4) if orig_h > 0 else "",
        })
        rows.append(row)

    return rect_bgr, warped_id_np, rows


def draw_overlay(img_bgr: np.ndarray, rect_bgr: np.ndarray, boxes: np.ndarray,
                 warped_id_np: np.ndarray, rows: List[Dict]) -> np.ndarray:
    """Sandingan kiri (asli+GT hijau) | kanan (rectified+recovered cyan / cropped merah)."""
    left = img_bgr.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(left, (int(round(x1)), int(round(y1))),
                      (int(round(x2)), int(round(y2))), (0, 255, 0), 2)

    right = rect_bgr.copy()
    ori_h, ori_w = right.shape[:2]
    for i, row in enumerate(rows):
        if row.get("dropped"):
            continue
        ys, xs = np.where(warped_id_np == (i + 1))
        if xs.size == 0:
            continue
        rx1, ry1, rx2, ry2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        color = (0, 0, 255) if row.get("cropped") else (255, 255, 0)  # merah / cyan (BGR)
        cv2.rectangle(right, (rx1, ry1), (rx2, ry2), color, 2)

    h = min(left.shape[0], right.shape[0])
    lw = int(left.shape[1] * h / left.shape[0])
    rw = int(right.shape[1] * h / right.shape[0])
    left = cv2.resize(left, (lw, h))
    right = cv2.resize(right, (rw, h))
    sep = np.full((h, 4, 3), 255, np.uint8)
    combo = np.hstack([left, sep, right])
    cv2.putText(combo, "asli + GT (hijau)", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(combo, "rectified: recovered (cyan) / cropped (merah)", (lw + 12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return combo


def _rate(flags: List[int]) -> Optional[float]:
    return (sum(flags) / len(flags)) if flags else None


def aggregate(ds_id: str, display: str, in_domain: bool,
              all_rows: List[Dict], n_images: int) -> Dict:
    """Ringkas metrik per-box menjadi agregat per-dataset."""
    n_boxes = len(all_rows)
    dropped = [int(r["dropped"]) for r in all_rows]
    surviving = [r for r in all_rows if not r["dropped"]]

    def col(name: str) -> List[float]:
        return [float(r[name]) for r in surviving if r.get(name) not in ("", None)]

    widen_w = col("widen_w")
    widen_h = col("widen_h")
    # % box yang melebar/menyusut (salah satu dimensi menyimpang > EPS).
    widened, shrunk = [], []
    for r in surviving:
        ww, wh = r.get("widen_w"), r.get("widen_h")
        if ww in ("", None) or wh in ("", None):
            continue
        ww, wh = float(ww), float(wh)
        widened.append(int(ww > 1 + EPS or wh > 1 + EPS))
        shrunk.append(int(ww < 1 - EPS or wh < 1 - EPS))

    return {
        "id": ds_id,
        "display": display,
        "in_domain": in_domain,
        "n_images": n_images,
        "n_boxes": n_boxes,
        "n_dropped": sum(dropped),
        "drop_rate": _rate(dropped),
        "crop_rate": _rate([int(r["cropped"]) for r in surviving]),
        "widen_rate": _rate(widened),
        "shrink_rate": _rate(shrunk),
        "widen_w_mean": mean(widen_w),
        "widen_w_median": median(widen_w),
        "widen_h_mean": mean(widen_h),
        "widen_h_median": median(widen_h),
        "fill_ratio_mean": mean(col("fill_ratio")),
        "fill_ratio_p05": percentile(col("fill_ratio"), 0.05),
        "area_ratio_mean": mean(col("area_ratio")),
        "area_ratio_p05": percentile(col("area_ratio"), 0.05),
        # Sanity recovered-vs-flowwarp (harusnya ~1).
        "width_ratio_mean": mean(col("width_ratio")),
        "height_ratio_mean": mean(col("height_ratio")),
        "iou_mean": mean(col("iou")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verifikasi integritas geometris bbox setelah rektifikasi MOWA.")
    ap.add_argument("--mowa-root", type=Path, default=ROOT / "vendor" / "MOWA")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "vendor" / "MOWA" / "checkpoint" / "mowa_pretrained.pth")
    ap.add_argument("--limit", type=int, default=30, help="Maks gambar per dataset (0 = semua).")
    ap.add_argument("--datasets", default="",
                    help="Subset id dipisah koma (default: semua).")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--tau", type=float, default=0.6,
                    help="Ambang area_ratio; di bawah ini box dianggap ter-crop.")
    ap.add_argument("--tps-cap", type=int, default=384,
                    help="Sisi terpanjang komputasi TPS coarse (default 384).")
    ap.add_argument("--no-fp16", action="store_true", help="Nonaktifkan autocast fp16.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: MOWA butuh CUDA (utils_transform.resample hardcode .cuda()). "
              "Tidak ada GPU terdeteksi — jalankan di mesin ber-GPU.", file=sys.stderr)
        return 3

    device = torch.device(f"cuda:{args.gpu}")
    use_fp16 = not args.no_fp16

    wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
    targets = [d for d in DATASETS if not wanted or d["id"] in wanted]
    if not targets:
        print(f"ERROR: tidak ada dataset cocok dengan --datasets={args.datasets}", file=sys.stderr)
        return 2

    add_mowa_to_path(args.mowa_root)
    print(f"[verify_bbox] device={device}")
    net = build_net(device)
    load_checkpoint(net, args.checkpoint, device)
    net.eval()
    print(f"[verify_bbox] model dimuat dari {args.checkpoint}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_dataset: List[Dict] = []

    for ds in targets:
        pairs = collect_pairs(ds, args.limit)
        if not pairs:
            print(f"[{ds['id']}] tak ada gambar — lewati.")
            per_dataset.append({
                "id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
                "status": "no_images", "n_images": 0, "n_boxes": 0,
            })
            continue

        overlay_dir = OUT_DIR / f"{ds['id']}_overlays"
        overlay_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[Dict] = []
        n_saved_overlay = 0
        n_images = 0
        for j, (img_path, lbl_path) in enumerate(pairs, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [{ds['id']}] SKIP tak terbaca: {img_path.name}", file=sys.stderr)
                continue
            ori_h, ori_w = img.shape[:2]
            classes, boxes = read_yolo_labels(lbl_path, ori_w, ori_h)
            n_images += 1
            if len(boxes) == 0:
                continue

            try:
                rect_bgr, warped_id_np, rows = process_image(
                    net, img, boxes, device, args.tps_cap, args.tau, use_fp16)
            except Exception as e:  # noqa: BLE001
                print(f"  [{ds['id']}] GAGAL {img_path.name}: {e}", file=sys.stderr)
                continue

            for r in rows:
                r["image"] = img_path.name
            all_rows.extend(rows)

            if n_saved_overlay < OVERLAYS_PER_DATASET:
                combo = draw_overlay(img, rect_bgr, boxes, warped_id_np, rows)
                cv2.imwrite(str(overlay_dir / f"{img_path.stem}_overlay.png"), combo)
                n_saved_overlay += 1

            if j % 10 == 0 or j == len(pairs):
                print(f"  [{ds['id']}] {j}/{len(pairs)} gambar, {len(all_rows)} box terkumpul")

        # CSV per-box.
        csv_fields = [
            "image", "box_idx", "orig_w", "orig_h", "flowwarp_w", "flowwarp_h",
            "recovered_w", "recovered_h", "filled_px", "fill_ratio", "area_ratio",
            "width_ratio", "height_ratio", "iou", "widen_w", "widen_h",
            "touches_border", "cropped", "dropped",
        ]
        write_csv(OUT_DIR / f"{ds['id']}_metrics.csv", all_rows, fieldnames=csv_fields)

        agg = aggregate(ds["id"], ds["display"], ds["in_domain"], all_rows, n_images)
        agg["status"] = "ok"
        per_dataset.append(agg)
        print(f"[{ds['id']}] boxes={agg['n_boxes']} dropped={agg['n_dropped']} "
              f"crop_rate={agg['crop_rate']} widen_rate={agg['widen_rate']} "
              f"shrink_rate={agg['shrink_rate']} widen_w_med={agg['widen_w_median']}")

    # Agregat keseluruhan (gabung semua box dataset yang ok).
    ok_ds = [d for d in per_dataset if d.get("status") == "ok"]
    overall = None
    if ok_ds:
        tot_boxes = sum(d["n_boxes"] for d in ok_ds)
        tot_dropped = sum(d["n_dropped"] for d in ok_ds)
        overall = {
            "n_datasets": len(ok_ds),
            "n_boxes": tot_boxes,
            "n_dropped": tot_dropped,
            "drop_rate": (tot_dropped / tot_boxes) if tot_boxes else None,
            # rata-rata sederhana antar-dataset (setiap dataset bobot sama).
            "crop_rate_macro": mean([d["crop_rate"] for d in ok_ds if d["crop_rate"] is not None]),
            "widen_rate_macro": mean([d["widen_rate"] for d in ok_ds if d["widen_rate"] is not None]),
            "shrink_rate_macro": mean([d["shrink_rate"] for d in ok_ds if d["shrink_rate"] is not None]),
        }

    summary = {
        "checkpoint": str(args.checkpoint),
        "tau": args.tau,
        "tps_cap": args.tps_cap,
        "fp16": use_fp16,
        "limit": args.limit,
        "eps_widen_shrink": EPS,
        "metric_notes": {
            "widen_w/widen_h": "flowwarp_dim / orig_dim; >1 melebar, <1 menyusut (pertanyaan inti).",
            "width_ratio/height_ratio/iou": "recovered (kanvas id) vs flowwarp; sanity ~1.",
            "fill_ratio": "pixel_terisi / (rec_w*rec_h); turun jika kotak melengkung/miring.",
            "area_ratio": "pixel_terisi / area_flowwarp; turun jika ter-crop keluar frame.",
            "cropped": "menyentuh tepi frame ATAU area_ratio < tau.",
            "dropped": "kotak hilang total (ter-warp keluar frame).",
        },
        "datasets": per_dataset,
        "overall": overall,
    }
    write_json(OUT_DIR / "bbox_integrity_summary.json", summary)
    print(f"[verify_bbox] tulis {OUT_DIR / 'bbox_integrity_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
