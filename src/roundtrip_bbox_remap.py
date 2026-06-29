"""
roundtrip_bbox_remap.py — Petakan deteksi dari frame RECTIFIED balik ke frame ASLI
(inverse warp MOWA), lalu bandingkan mAP A vs B.

Tujuan (skripsi):
  Pola standar "detect-in-rectified, report-in-original": jalankan YOLO pada gambar
  hasil MOWA (rectified), lalu PETAKAN bbox prediksi kembali ke koordinat gambar asli
  (distorsi) memakai inverse flow MOWA. Bandingkan dua skor mAP:
    (A) rectified-space  : bbox prediksi (frame rectified) vs GT yang sudah di-warp
                           (data/rectified/<id>/labels, sudah selaras frame rectified).
    (B) original-space   : bbox prediksi yang DIPETAKAN BALIK ke frame asli vs GT ASLI
                           (data/images/val atau data/external/<id>/.../labels).
  Ini menjawab: apakah "deteksi di rectified lalu petakan balik" memulihkan akurasi
  dibanding deteksi langsung di gambar asli.

DUA MODE (karena venv MOWA dan venv YOLO BERBEDA):
  Perhitungan flow MOWA butuh .venv-mowa; predict YOLO butuh .venv-yolo. Karena itu
  kerja DIPISAH menjadi dua tahap yang dijalankan berurutan:

  URUTAN JALAN:
    1) Tahap flow (di .venv-mowa) — hitung & cache inverse map per gambar:
         .venv-mowa/Scripts/python.exe src/roundtrip_bbox_remap.py --flows-only \
             --mowa-root vendor/MOWA \
             --checkpoint vendor/MOWA/checkpoint/mowa_pretrained.pth \
             --datasets all --limit 0

    2) Tahap deteksi+remap+eval (di .venv-yolo) — predict, petakan balik, skor:
         .venv-yolo/Scripts/python.exe src/roundtrip_bbox_remap.py \
             --weights "train model/runs_rectified/ft_rectified_yolov8m/weights/best.pt" \
             --datasets all --limit 0

  Mode --flows-only WAJIB dijalankan lebih dulu (di .venv-mowa). Mode default akan
  berhenti dengan pesan jelas bila cache inverse map belum ada.

INVERSE MAP (apa yang di-cache):
  Untuk tiap gambar disimpan peta M berbentuk (H, W, 2): untuk tiap pixel (x, y) di
  frame RECTIFIED, M[y, x] = (x_asli, y_asli) di gambar ASLI. M dihitung dengan
  menerapkan warp penuh MOWA (tps2flow lalu flow — sama persis dengan yang dipakai
  mowa_rectify untuk membuat gambar rectified) pada GRID IDENTITAS:
      M[p] = _apply_full_warp(identity_grid)[p] = (p + flow[p]) + tps2flow[p + flow[p]]
  yaitu koordinat sumber (frame asli) yang di-sampel untuk mengisi pixel rectified p.
  Jadi M adalah pemetaan rectified->asli yang tepat untuk memetakan balik bbox.

  Peta disimpan sebagai .npz float16 (np.savez_compressed) di
      reports/roundtrip_bbox/_flows/<id>/<stem>.npz
  CAVEAT UKURAN: ~ H*W*2*2 byte (mis. 1920x1080 ~ 8 MB/gambar). File ini BESAR dan
  di-.gitignore (regenerasi via --flows-only). CAVEAT PRESISI: float16 punya ULP ~1px
  pada magnitudo ~1024-2048; untuk koordinat < 2048 (resolusi umum) galat <= ~1px,
  dapat diabaikan untuk pencocokan bbox IoU@0.5.

METRIK (AP50, single class):
  AP50 dihitung sendiri (self-contained) dengan interpolasi all-point (VOC2010+/COCO
  area): prediksi diurutkan turun berdasar confidence, dicocokkan greedy ke GT dengan
  IoU >= --iou-thr (tiap GT dipakai sekali), lalu AP = luas di bawah selubung
  precision-recall. Pooled per dataset (akumulasi seluruh gambar) dan overall.

Output:
  reports/roundtrip_bbox/<id>_remap.csv       per-gambar: jumlah box, mean IoU, AP50 A vs B
  reports/roundtrip_bbox/roundtrip_bbox_summary.json  per-dataset + overall A vs B
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Skrip di src/ dijalankan langsung, jadi folder src/ ada di sys.path[0].
from mowa_rectify import (
    add_mowa_to_path,
    build_net,
    load_checkpoint,
    compute_flows,
    _apply_full_warp,
    _identity_grid_xy,
    read_yolo_labels,
    list_images,
    INPUT_SIZE,
)
from eval_detection import DATASETS, resolve_val_dirs

import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "roundtrip_bbox"
FLOWS_DIR = OUT_DIR / "_flows"
DEFAULT_WEIGHTS = ROOT / "train model" / "runs_rectified" / "ft_rectified_yolov8m" / "weights" / "best.pt"
TPS_POINTS = [10, 12, 14, 16]


# ---------------------------------------------------------------------------
# Pemilihan dataset
# ---------------------------------------------------------------------------
def select_datasets(spec: str) -> List[Dict]:
    """Filter DATASETS berdasar --datasets ('all' atau daftar id dipisah koma)."""
    if not spec or spec.strip().lower() == "all":
        return list(DATASETS)
    ids = {s.strip() for s in spec.split(",") if s.strip()}
    chosen = [d for d in DATASETS if d["id"] in ids]
    if not chosen:
        known = ", ".join(d["id"] for d in DATASETS)
        raise SystemExit(f"ERROR: --datasets '{spec}' tak cocok. Pilihan: {known}")
    return chosen


# ---------------------------------------------------------------------------
# MODE --flows-only : hitung + cache inverse map (rectified -> asli). Butuh .venv-mowa.
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_inverse_map(net, img_bgr: np.ndarray, device: torch.device,
                        use_fp16: bool, tps_cap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Peta inverse + mask validitas.

    Return (M, V):
      M : (H, W, 2) float32 — untuk pixel rectified (x, y), koordinat
          (x_asli, y_asli) di gambar ASLI.
      V : (H, W) bool — True bila pixel rectified itu MENGAMBIL sampel dari DALAM
          frame asli (bukan border hitam hasil warp).

    Diperoleh dengan menerapkan warp penuh MOWA (tps2flow lalu flow) pada grid
    identitas: hasil di pixel p adalah koordinat sumber src(p) yang di-sampel untuk
    membuat pixel rectified p — persis pemetaan rectified->asli yang dibutuhkan.

    CATATAN border: grid_sample memakai padding_mode='zeros', jadi pixel yang
    ter-warp DI LUAR frame asli menghasilkan M=(0,0). Nilai (0,0) itu tak bisa
    dibedakan dari koordinat sudut kiri-atas yang sah, sehingga min/max bbox bisa
    tertarik ke origin. Karena itu kita ikut menghitung V dengan mem-warp field
    "ones" (mode nearest): 1 => sampel dari dalam frame, 0 => border. remap hanya
    memakai pixel V=True.

    CAVEAT presisi tepi: M dihitung bilinear sedang V nearest, jadi tepat di batas
    valid/border satu pixel M bisa mencampur koordinat asli dgn 0 (bias ke origin
    <= ~1px) walau V=True. Efeknya terbatas pada box yang menyentuh tepi frame dan
    dapat diabaikan untuk IoU@0.5 (setara galat float16 yang sudah didokumentasikan).
    """
    ori_h, ori_w = img_bgr.shape[:2]
    resized = cv2.resize(img_bgr, (INPUT_SIZE, INPUT_SIZE)).astype(np.float32) / 255.0
    input2 = np.transpose(resized, (2, 0, 1))[None]  # 1,3,256,256
    mask = np.ones((1, 1, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
    input2_t = torch.from_numpy(input2).float().to(device)
    mask_t = torch.from_numpy(mask).float().to(device)

    tps2flow, flow = compute_flows(net, input2_t, mask_t, ori_h, ori_w, TPS_POINTS,
                                   device, use_fp16, tps_cap=tps_cap)

    identity = _identity_grid_xy(ori_h, ori_w, device)  # 1,2,H,W (channel 0=x, 1=y)
    warped = _apply_full_warp(identity, tps2flow, flow, mode="bilinear")[0]  # 2,H,W
    M = warped.permute(1, 2, 0).contiguous().float().cpu().numpy()  # H,W,2

    ones = torch.ones((1, 1, ori_h, ori_w), dtype=torch.float32, device=device)
    valid = _apply_full_warp(ones, tps2flow, flow, mode="nearest")[0, 0]  # H,W
    V = (valid.cpu().numpy() > 0.5)
    return M, V


def run_flows_only(args) -> int:
    if not torch.cuda.is_available():
        print("ERROR: mode --flows-only butuh CUDA (MOWA hardcode operasi .cuda()). "
              "Tidak ada GPU terdeteksi. Jalankan di mesin ber-GPU dengan .venv-mowa.",
              file=sys.stderr)
        return 3

    device = torch.device(f"cuda:{args.gpu}")
    add_mowa_to_path(args.mowa_root)
    print(f"[roundtrip:flows] device={device}")
    net = build_net(device)
    load_checkpoint(net, args.checkpoint, device)
    net.eval()
    print(f"[roundtrip:flows] model dimuat dari {args.checkpoint}")

    use_fp16 = not args.no_fp16
    datasets = select_datasets(args.datasets)
    total_saved, total_failed = 0, 0
    t0 = time.time()

    for ds in datasets:
        out_dir = FLOWS_DIR / ds["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        # Inverse map dihitung dari gambar ASLI (input MOWA), stem sama dgn rectified.
        # Pakai indeks stem->path yang SAMA (first-wins) dgn mode default supaya
        # (a) tak ada tabrakan stem antar-split (chicken_detection_fum punya 3 split)
        # dan (b) --limit memilih himpunan stem yang identik di kedua mode
        # (diurut per-stem, sama seperti list_images pada folder rectified gabungan).
        orig_index = _build_orig_index(ds)
        stems = sorted(orig_index.keys())
        if args.limit > 0:
            stems = stems[: args.limit]
        imgs = [orig_index[s] for s in stems]
        if not imgs:
            print(f"[roundtrip:flows] {ds['id']}: tidak ada gambar asli, lewati.")
            continue

        saved, failed = 0, 0
        for i, img_path in enumerate(imgs, 1):
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                print(f"  SKIP (tak terbaca): {img_path.name}", file=sys.stderr)
                continue
            try:
                M, V = compute_inverse_map(net, img, device, use_fp16, args.tps_cap)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  GAGAL inverse-map {img_path.name}: {e}", file=sys.stderr)
                continue
            np.savez_compressed(out_dir / (img_path.stem + ".npz"),
                                M=M.astype(np.float16), V=np.packbits(V))
            saved += 1
            if i % 50 == 0 or i == len(imgs):
                dt = time.time() - t0
                print(f"  [{ds['id']}] {i}/{len(imgs)} saved={saved} failed={failed} "
                      f"({dt:.1f}s, {dt / max(1, total_saved + saved):.3f}s/img)")
        total_saved += saved
        total_failed += failed
        print(f"[roundtrip:flows] {ds['id']}: saved={saved} failed={failed} -> {out_dir}")

    print(f"[roundtrip:flows] SELESAI: saved={total_saved} failed={total_failed} "
          f"dalam {time.time() - t0:.1f}s")
    print("[roundtrip:flows] Lanjut mode default di .venv-yolo untuk predict+remap+eval.")
    return 0 if total_failed == 0 else 1


# ---------------------------------------------------------------------------
# Remap bbox rectified -> asli via inverse map (dipakai mode default)
# ---------------------------------------------------------------------------
def remap_boxes_rectified_to_original(boxes_xyxy: np.ndarray, inverse_map: np.ndarray,
                                      valid_map: Optional[np.ndarray] = None
                                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Petakan bbox (frame rectified, pixel) -> bbox axis-aligned (frame asli).

    boxes_xyxy : Nx4 (x1,y1,x2,y2) pixel di frame RECTIFIED.
    inverse_map: (H, W, 2) -> (x_asli, y_asli) untuk tiap pixel rectified.
    valid_map  : (H, W) bool opsional; True bila pixel mengambil sampel dari DALAM
                 frame asli. Pixel border (False) DIABAIKAN saat min/max supaya
                 sentinel (0,0) dari padding zeros tidak menarik sudut box ke origin.

    Untuk tiap box diambil sub-region inverse_map di dalam box (hanya pixel valid),
    lalu min/max koordinat asli yang dipetakan -> box pembungkus di frame asli (robust
    terhadap distorsi, mirip warp_boxes_via_flow yang memakai min/max instance-mask).

    Return (mapped Nx4 float32, valid boolean N). Box degenerat / tanpa pixel valid
    -> valid=False (prediksi tak-terpetakan; pemanggil menghitungnya sebagai FP).
    """
    n = len(boxes_xyxy)
    mapped = np.zeros((n, 4), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    if n == 0:
        return mapped, valid
    H, W = inverse_map.shape[:2]
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        xi1 = max(0, min(W - 1, int(round(float(x1)))))
        yi1 = max(0, min(H - 1, int(round(float(y1)))))
        xi2 = max(0, min(W, int(round(float(x2)))))
        yi2 = max(0, min(H, int(round(float(y2)))))
        if xi2 <= xi1 or yi2 <= yi1:
            continue
        sub = inverse_map[yi1:yi2, xi1:xi2]  # (h,w,2)
        xs = sub[..., 0]
        ys = sub[..., 1]
        if valid_map is not None:
            m = valid_map[yi1:yi2, xi1:xi2]
            if not m.any():
                continue  # seluruh box jatuh di border -> tak terpetakan
            xs = xs[m]
            ys = ys[m]
        ox1, ox2 = float(xs.min()), float(xs.max())
        oy1, oy2 = float(ys.min()), float(ys.max())
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        mapped[i] = (ox1, oy1, ox2, oy2)
        valid[i] = True
    return mapped, valid


# ---------------------------------------------------------------------------
# Metrik AP50 self-contained (single class)
# ---------------------------------------------------------------------------
def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU (Na, Nb) untuk dua set bbox xyxy pixel."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = boxes_a.astype(np.float64)
    b = boxes_b.astype(np.float64)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def match_preds(pred_boxes: np.ndarray, pred_conf: np.ndarray, gt_boxes: np.ndarray,
                iou_thr: float) -> Tuple[np.ndarray, List[float]]:
    """Cocokkan greedy prediksi (urut confidence turun) ke GT.

    Return (tp boolean sejajar pred_boxes, daftar IoU pasangan yang cocok).
    Tiap GT hanya boleh dicocokkan satu prediksi.
    """
    n = len(pred_boxes)
    tp = np.zeros(n, dtype=bool)
    matched_iou: List[float] = []
    if n == 0 or len(gt_boxes) == 0:
        return tp, matched_iou
    ious = iou_matrix(pred_boxes, gt_boxes)  # N,G
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    for i in np.argsort(-pred_conf):
        row = ious[i].copy()
        row[gt_used] = -1.0
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            tp[i] = True
            gt_used[j] = True
            matched_iou.append(float(ious[i, j]))
    return tp, matched_iou


def pool_ap(conf_list: List[np.ndarray], tp_list: List[np.ndarray], n_gt: int) -> float:
    """AP50 pooled: gabung seluruh prediksi (per-gambar) lalu hitung satu AP.

    conf_list/tp_list = daftar array per gambar; n_gt = total GT terkumpul.
    """
    if not conf_list:
        return 0.0
    return average_precision(np.concatenate(conf_list), np.concatenate(tp_list), n_gt)


def average_precision(conf: np.ndarray, tp: np.ndarray, n_gt: int) -> float:
    """AP50 interpolasi all-point (VOC2010+/COCO area). conf & tp sejajar per prediksi."""
    if n_gt == 0 or len(conf) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp_sorted = tp[order].astype(np.float64)
    fp_sorted = 1.0 - tp_sorted
    tp_cum = np.cumsum(tp_sorted)
    fp_cum = np.cumsum(fp_sorted)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


# ---------------------------------------------------------------------------
# MODE default : predict YOLO di rectified -> remap -> eval A vs B. Butuh .venv-yolo.
# ---------------------------------------------------------------------------
def _to_label_path(img_path: Path) -> Path:
    """Konvensi Ultralytics: ganti komponen 'images' -> 'labels', suffix .txt."""
    parts = list(img_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _build_orig_index(ds: Dict) -> Dict[str, Path]:
    """Map stem -> path gambar ASLI (untuk menemukan label GT asli)."""
    idx: Dict[str, Path] = {}
    for d in resolve_val_dirs(ds, None):
        d = Path(d)
        if d.is_dir():
            for p in list_images(d):
                idx.setdefault(p.stem, p)
    return idx


def evaluate_dataset(model, ds: Dict, args) -> Tuple[Dict, List[List]]:
    """Predict tiap gambar rectified, remap, dan skor A vs B untuk satu dataset.

    Return (ringkasan dataset, baris CSV per-gambar).
    """
    flows_dir = FLOWS_DIR / ds["id"]
    # Folder rectified pakai rectified_subdir (konvensi eval_detection.resolve_val_dirs),
    # bukan id, agar konsisten dgn mowa_rectify/eval walau kelak subdir != id.
    rect_img_dir = Path(resolve_val_dirs(ds, ROOT / "data" / "rectified")[0])
    rect_lbl_dir = rect_img_dir.parent / "labels"

    if not flows_dir.is_dir():
        return ({"id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
                 "status": "no_flows",
                 "hint": f"cache inverse map tak ada di {flows_dir}. Jalankan dulu "
                         f"--flows-only di .venv-mowa."}, [])
    if not rect_img_dir.is_dir():
        return ({"id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
                 "status": "no_rectified_images",
                 "hint": f"gambar rectified tak ada di {rect_img_dir}. Jalankan mowa_rectify."}, [])

    # Urutkan per-stem (bukan per-nama-file) agar --limit memilih himpunan stem yang
    # IDENTIK dengan flows-only (yang juga stem-sorted). list_images mengurut per nama
    # file lengkap sehingga ekstensi/pemisah bisa menggeser urutan untuk stem prefiks.
    imgs = sorted(list_images(rect_img_dir), key=lambda p: p.stem)
    if args.limit > 0:
        imgs = imgs[: args.limit]
    if not imgs:
        return ({"id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
                 "status": "no_images"}, [])

    orig_index = _build_orig_index(ds)

    # Akumulator pooled untuk AP50 tingkat dataset.
    conf_A, tp_A, ngt_A = [], [], 0
    conf_B, tp_B, ngt_B = [], [], 0
    iou_all_A: List[float] = []
    iou_all_B: List[float] = []
    n_pred_total = 0
    n_unmapped_total = 0
    missing_flows = 0
    missing_orig = 0
    rows: List[List] = []

    for img_path in imgs:
        stem = img_path.stem
        npz_path = flows_dir / (stem + ".npz")
        if not npz_path.exists():
            missing_flows += 1
            continue
        with np.load(npz_path) as z:  # context manager: tutup handle .npz
            M = z["M"].astype(np.float32)  # H,W,2 (dims frame asli)
            ori_h, ori_w = M.shape[:2]
            V = np.unpackbits(z["V"], count=ori_h * ori_w).reshape(ori_h, ori_w).astype(bool) \
                if "V" in z else None

        res = model.predict(str(img_path), imgsz=args.imgsz, conf=args.conf,
                            device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            pboxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)  # pixel frame rectified
            pconf = res.boxes.conf.cpu().numpy().astype(np.float32)
        else:
            pboxes = np.zeros((0, 4), dtype=np.float32)
            pconf = np.zeros((0,), dtype=np.float32)
        rect_h, rect_w = res.orig_shape  # dimensi gambar rectified
        n_pred_total += len(pboxes)

        # (A) rectified-space: prediksi vs GT warp (data/rectified/<id>/labels).
        _, gtA = read_yolo_labels(rect_lbl_dir / (stem + ".txt"), rect_w, rect_h)
        tpA, iouA = match_preds(pboxes, pconf, gtA, args.iou_thr)
        apA = average_precision(pconf, tpA, len(gtA))

        # (B) original-space: prediksi dipetakan balik vs GT ASLI.
        # Semua prediksi TETAP diskor: box yang gagal dipetakan (keep=False) bernilai
        # (0,0,0,0) sehingga IoU=0 dgn semua GT -> tak pernah cocok -> dihitung FP.
        # Ini menjaga keadilan A vs B (A juga menskor seluruh prediksi).
        mapped, keep = remap_boxes_rectified_to_original(pboxes, M, V)
        n_unmapped_total += int((~keep).sum())
        if stem in orig_index:
            _, gtB = read_yolo_labels(_to_label_path(orig_index[stem]), ori_w, ori_h)
        else:
            missing_orig += 1
            gtB = np.zeros((0, 4), dtype=np.float32)
        tpB, iouB = match_preds(mapped, pconf, gtB, args.iou_thr)
        apB = average_precision(pconf, tpB, len(gtB))

        # Akumulasi pooled.
        conf_A.append(pconf); tp_A.append(tpA); ngt_A += len(gtA)
        conf_B.append(pconf); tp_B.append(tpB); ngt_B += len(gtB)
        iou_all_A.extend(iouA); iou_all_B.extend(iouB)

        rows.append([
            stem, len(pboxes), len(gtA), len(gtB),
            round(float(np.mean(iouA)) if iouA else 0.0, 5),
            round(float(np.mean(iouB)) if iouB else 0.0, 5),
            round(apA, 5), round(apB, 5),
        ])

    ap50_A = pool_ap(conf_A, tp_A, ngt_A)
    ap50_B = pool_ap(conf_B, tp_B, ngt_B)
    summary = {
        "id": ds["id"], "display": ds["display"], "in_domain": ds["in_domain"],
        "status": "ok",
        "images": len(rows),
        "n_pred": n_pred_total,
        "n_unmapped_pred": n_unmapped_total,
        "n_gt_rect": ngt_A,
        "n_gt_orig": ngt_B,
        "ap50_A_rectified": round(ap50_A, 5),
        "ap50_B_mapped_original": round(ap50_B, 5),
        "delta_B_minus_A": round(ap50_B - ap50_A, 5),
        "mean_iou_A": round(float(np.mean(iou_all_A)) if iou_all_A else 0.0, 5),
        "mean_iou_B": round(float(np.mean(iou_all_B)) if iou_all_B else 0.0, 5),
        "missing_flows": missing_flows,
        "missing_orig_labels": missing_orig,
        # data mentah pooled untuk overall (tidak ditulis ke JSON akhir).
        "_conf_A": conf_A, "_tp_A": tp_A, "_ngt_A": ngt_A,
        "_conf_B": conf_B, "_tp_B": tp_B, "_ngt_B": ngt_B,
    }
    return summary, rows


def run_default(args) -> int:
    if not args.weights.exists():
        print(f"ERROR: weights tidak ditemukan: {args.weights}", file=sys.stderr)
        return 2
    if not FLOWS_DIR.is_dir():
        print(f"ERROR: cache inverse map belum ada di {FLOWS_DIR}.\n"
              f"Jalankan dulu tahap flow di .venv-mowa:\n"
              f"  .venv-mowa/Scripts/python.exe src/roundtrip_bbox_remap.py --flows-only ...",
              file=sys.stderr)
        return 4

    from ultralytics import YOLO

    args.device = str(args.gpu) if torch.cuda.is_available() else "cpu"
    print(f"[roundtrip:eval] weights={args.weights} device={args.device} "
          f"imgsz={args.imgsz} conf={args.conf} iou_thr={args.iou_thr}")
    model = YOLO(str(args.weights))

    datasets = select_datasets(args.datasets)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_summaries: List[Dict] = []

    g_conf_A, g_tp_A, g_ngt_A = [], [], 0
    g_conf_B, g_tp_B, g_ngt_B = [], [], 0

    for ds in datasets:
        print(f"[roundtrip:eval] {ds['id']} ...")
        summary, rows = evaluate_dataset(model, ds, args)

        if summary.get("status") == "ok":
            # Tulis CSV per dataset.
            csv_path = OUT_DIR / f"{ds['id']}_remap.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["stem", "n_pred", "n_gt_rect", "n_gt_orig",
                            "mean_iou_A", "mean_iou_B", "ap50_A", "ap50_B"])
                w.writerows(rows)
            print(f"   AP50 A(rectified)={summary['ap50_A_rectified']:.4f}  "
                  f"B(mapped->orig)={summary['ap50_B_mapped_original']:.4f}  "
                  f"delta={summary['delta_B_minus_A']:+.4f}  (n={summary['images']}) -> {csv_path.name}")
            if summary["missing_flows"]:
                print(f"   catatan: {summary['missing_flows']} gambar tanpa cache flow (dilewati).")
            # Akumulasi overall.
            g_conf_A += summary.pop("_conf_A"); g_tp_A += summary.pop("_tp_A"); g_ngt_A += summary.pop("_ngt_A")
            g_conf_B += summary.pop("_conf_B"); g_tp_B += summary.pop("_tp_B"); g_ngt_B += summary.pop("_ngt_B")
        else:
            print(f"   SKIP ({summary.get('status')}): {summary.get('hint', '')}")
            for k in ("_conf_A", "_tp_A", "_ngt_A", "_conf_B", "_tp_B", "_ngt_B"):
                summary.pop(k, None)
        ds_summaries.append(summary)

    overall_A = pool_ap(g_conf_A, g_tp_A, g_ngt_A)
    overall_B = pool_ap(g_conf_B, g_tp_B, g_ngt_B)
    payload = {
        "condition": "roundtrip_bbox_remap",
        "weights": str(args.weights),
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou_thr": args.iou_thr,
        "metric": ("AP50 interpolasi all-point (VOC2010+/COCO area), single class, "
                   "IoU>=iou_thr. A = prediksi frame rectified vs GT warp "
                   "(data/rectified/<id>/labels). B = prediksi dipetakan balik ke frame "
                   "asli via inverse map MOWA vs GT ASLI."),
        "datasets": ds_summaries,
        "overall": {
            "ap50_A_rectified": round(overall_A, 5),
            "ap50_B_mapped_original": round(overall_B, 5),
            "delta_B_minus_A": round(overall_B - overall_A, 5),
        },
    }
    summary_path = OUT_DIR / "roundtrip_bbox_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[roundtrip:eval] overall AP50 A={overall_A:.4f} B={overall_B:.4f} "
          f"delta={overall_B - overall_A:+.4f}")
    print(f"[roundtrip:eval] tulis {summary_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Roundtrip bbox remap rectified->asli via inverse warp MOWA; "
                    "bandingkan mAP A (rectified) vs B (mapped->original).")
    ap.add_argument("--flows-only", action="store_true",
                    help="Tahap flow (di .venv-mowa): hitung & cache inverse map. Jalankan DULU.")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="Bobot YOLO (default ft-rectified best.pt). Mode default.")
    ap.add_argument("--datasets", default="all",
                    help="'all' atau daftar id dipisah koma (pio_val,broiler_instance_seg,...).")
    ap.add_argument("--limit", type=int, default=0, help="Proses N gambar pertama tiap dataset (0=semua).")
    ap.add_argument("--gpu", type=int, default=0, help="Indeks GPU (cuda:<gpu>).")
    ap.add_argument("--mowa-root", type=Path, default=ROOT / "vendor" / "MOWA", help="Root repo MOWA.")
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "vendor" / "MOWA" / "checkpoint" / "mowa_pretrained.pth")
    ap.add_argument("--iou-thr", type=float, default=0.5, help="Ambang IoU pencocokan AP50.")
    ap.add_argument("--conf", type=float, default=0.25, help="Ambang confidence predict YOLO.")
    ap.add_argument("--imgsz", type=int, default=960, help="imgsz predict YOLO (samakan dgn eval_detection).")
    ap.add_argument("--tps-cap", type=int, default=384, help="Sisi terpanjang komputasi TPS coarse (flows-only).")
    ap.add_argument("--no-fp16", action="store_true", help="Nonaktifkan autocast fp16 (flows-only).")
    args = ap.parse_args()

    if args.flows_only:
        return run_flows_only(args)
    return run_default(args)


if __name__ == "__main__":
    raise SystemExit(main())
