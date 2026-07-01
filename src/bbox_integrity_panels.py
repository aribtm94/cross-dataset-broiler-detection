"""
bbox_integrity_panels.py — Cek integritas geometri bounding box setelah rektifikasi MOWA.

Tujuan (skripsi):
  Pembimbing ingin memverifikasi SECARA VISUAL bahwa bounding box setelah rektifikasi
  MOWA masih benar secara geometris — tidak melebar / menyempit tak wajar, dan ayam
  tidak terpotong keluar frame ("kotak hitam" pinggir gambar hasil warp). Skrip ini
  membuat panel 3-kolom per gambar sampel:

    Kolom 1 — ASLI      : gambar asli + GT (hijau), dibaca dari label YOLO asli.
    Kolom 2 — RECTIFIED : gambar hasil MOWA + box hasil WARP (oranye), dibaca dari
                          label rectified yang sudah ditransformasi flow MOWA.
    Kolom 3 — INTEGRITAS: gambar rectified + overlay MASK HITAM (magenta, wilayah yang
                          ter-warp keluar frame) + box "pulih" (merah) = box warp yang
                          sama, PLUS anotasi Δlebar% & Δtinggi% (box warp vs GT asli).

  Unit ini TIDAK butuh GPU / MOWA. Ia hanya membaca gambar + label yang SUDAH dibuat
  lebih dulu oleh `src/mowa_rectify.py --label-mode warp`, dari disk:
    data/rectified/<id>/{images,labels}   (hasil rektifikasi + box warp)
    label & gambar ASLI                    (lihat DATASET_SPECS di bawah)

  Catatan pemulihan box: pemulihan box "benar" dari region hitam adalah milik unit lain
  (Task 1 recovery). Di sini kolom 3 mengaproksimasi box pulih = box warp (merah) di atas
  mask hitam, cukup untuk mata pembimbing menilai apakah ayam terpotong.

Kecocokan GT↔warp:
  Label warp menyimpan SUBSET GT (box yang keluar frame dibuang) tanpa indeks asli, jadi:
    - jika jumlah GT == jumlah warp  -> cocokkan per-urutan baris (indeks).
    - jika berbeda                   -> cocokkan IoU-greedy pada ruang koordinat
      ternormalisasi (rektifikasi MOWA moderat, IoU ternormalisasi jadi proksi wajar).

Konvensi: satu kelas (id 0). ROOT = parents[1]. Output PNG (berat) TIDAK di-commit.

Contoh (murni OpenCV, tanpa GPU — venv apa pun yang punya cv2/numpy):
  .venv-yolo/Scripts/python.exe src/bbox_integrity_panels.py
  .venv-yolo/Scripts/python.exe src/bbox_integrity_panels.py --datasets pio_val --limit 12
  .venv-yolo/Scripts/python.exe src/bbox_integrity_panels.py --datasets chicken_detection_fum --limit 16 --cols 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RECT_ROOT = ROOT / "data" / "rectified"
OUT_ROOT = ROOT / "reports" / "bbox_integrity_panels"

# ---- warna (BGR untuk cv2) ----
GT_COLOR = (0, 200, 60)        # hijau  — GT asli (kolom 1)
WARP_COLOR = (0, 150, 240)     # oranye — box warp (kolom 2)
RECOVER_COLOR = (0, 0, 235)    # merah  — box "pulih" (kolom 3)
MASK_TINT = (255, 0, 200)      # magenta — overlay region hitam
CAP_BG = (38, 30, 24)          # bar caption gelap
HDR_BG = (67, 42, 15)          # navy-ish header (BGR)
WHITE = (255, 255, 255)
MUTED = (190, 190, 190)
BORDER = (60, 60, 60)

CELL_W = 640          # lebar tiap kolom gambar (px)
GAP = 10              # sela antar kolom
CAP_H = 62           # tinggi bar caption per kolom
HDR_H = 46           # tinggi header panel
FONT = cv2.FONT_HERSHEY_SIMPLEX
PERBOX_MAX = 20      # jika jumlah box tercocok <= ini, tulis Δ% per-box; kalau padat ringkas saja
BLACK_THRESH = 10    # piksel dianggap "hitam" jika semua kanal < ini


# --------------------------------------------------------------------------------------
# Definisi dataset: rect_sub -> gambar/label ASLI (cari stem di semua dir)
# --------------------------------------------------------------------------------------
def dataset_specs() -> Dict[str, Dict]:
    fum = ROOT / "data" / "external" / "chicken_detection_fum"
    bis = ROOT / "data" / "external" / "broiler_instance_seg" / "train"
    return {
        "pio_val": {
            "display": "PIO val (in-domain)",
            "orig_img_dirs": [ROOT / "data" / "images" / "val"],
            "orig_lbl_dirs": [ROOT / "data" / "labels" / "val"],
        },
        "broiler_instance_seg": {
            "display": "Roboflow broiler_instance_seg (external)",
            "orig_img_dirs": [bis / "images"],
            "orig_lbl_dirs": [bis / "labels"],
        },
        "chicken_detection_fum": {
            "display": "Roboflow chicken_detection_fum (external)",
            "orig_img_dirs": [fum / "test" / "images", fum / "valid" / "images", fum / "train" / "images"],
            "orig_lbl_dirs": [fum / "test" / "labels", fum / "valid" / "labels", fum / "train" / "labels"],
        },
    }


def _index_dir(dirs: List[Path], suffixes) -> Dict[str, Path]:
    """Peta stem -> path pertama yang ditemukan di daftar folder."""
    out: Dict[str, Path] = {}
    for d in dirs:
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() in suffixes:
                    out.setdefault(p.stem, p)
    return out


def build_pairs(ds_id: str, spec: Dict) -> Tuple[List[Dict], int]:
    """Kumpulkan pasangan (asli, rectified) untuk satu dataset.

    Iterasi atas gambar RECTIFIED (itu yang tersedia); cocokkan ke asli via stem.
    Return (entries, n_skip) di mana n_skip = gambar rectified tanpa pasangan asli.
    """
    rect_img_dir = RECT_ROOT / ds_id / "images"
    rect_lbl_dir = RECT_ROOT / ds_id / "labels"
    if not rect_img_dir.is_dir():
        return [], 0

    orig_img_map = _index_dir(spec["orig_img_dirs"], IMAGE_EXTS)
    orig_lbl_map = _index_dir(spec["orig_lbl_dirs"], {".txt"})

    entries: List[Dict] = []
    n_skip = 0
    for rect_img in sorted(rect_img_dir.iterdir()):
        if rect_img.suffix.lower() not in IMAGE_EXTS:
            continue
        stem = rect_img.stem
        orig_img = orig_img_map.get(stem)
        if orig_img is None:
            n_skip += 1  # tak ada gambar asli pasangannya
        entries.append({
            "stem": stem,
            "orig_img": orig_img,
            "orig_lbl": orig_lbl_map.get(stem),
            "rect_img": rect_img,
            "rect_lbl": rect_lbl_dir / (stem + ".txt"),
        })
    return entries, n_skip


# --------------------------------------------------------------------------------------
# IO gambar aman-Unicode (Windows) + label YOLO
# --------------------------------------------------------------------------------------
def imread(path: Path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite(path: Path, img: np.ndarray) -> bool:
    try:
        ok, buf = cv2.imencode(path.suffix or ".png", img)
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False


def read_yolo_norm(path: Optional[Path]) -> List[Tuple[float, float, float, float]]:
    """Baca label YOLO -> list (cx,cy,w,h) ternormalisasi. Kelas diabaikan (semua id 0)."""
    rows: List[Tuple[float, float, float, float]] = []
    if path is None or not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        except ValueError:
            continue
        rows.append((cx, cy, w, h))
    return rows


def norm_to_xyxy(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[float, float, float, float]:
    cx, cy, bw, bh = box
    return ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)


# --------------------------------------------------------------------------------------
# Kecocokan GT (asli) <-> box warp (rectified) + Δ% ukuran
# --------------------------------------------------------------------------------------
def _iou_norm(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """IoU dua box (cx,cy,w,h) ternormalisasi."""
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = a[2] * a[3] + b[2] * b[3] - inter
    return inter / ua if ua > 0 else 0.0


def match_gt_warp(gt: List[Tuple[float, float, float, float]],
                  warp: List[Tuple[float, float, float, float]]) -> List[Tuple[int, int]]:
    """Cocokkan GT<->warp. Sama jumlah -> per-indeks; beda -> IoU-greedy. Return list (gi,wi)."""
    if not gt or not warp:
        return []
    if len(gt) == len(warp):
        return [(i, i) for i in range(len(gt))]
    # IoU-greedy
    cand: List[Tuple[float, int, int]] = []
    for gi, g in enumerate(gt):
        for wi, w in enumerate(warp):
            iou = _iou_norm(g, w)
            if iou > 0:
                cand.append((iou, gi, wi))
    cand.sort(reverse=True)
    used_g, used_w = set(), set()
    pairs: List[Tuple[int, int]] = []
    for iou, gi, wi in cand:
        if gi in used_g or wi in used_w:
            continue
        used_g.add(gi)
        used_w.add(wi)
        pairs.append((gi, wi))
    return pairs


def size_deltas(gt: List[Tuple[float, float, float, float]],
                warp: List[Tuple[float, float, float, float]],
                pairs: List[Tuple[int, int]]) -> Dict[int, Tuple[float, float]]:
    """Δ% lebar & tinggi (warp vs GT) per indeks warp: (dw%, dh%)."""
    out: Dict[int, Tuple[float, float]] = {}
    for gi, wi in pairs:
        gw, gh = gt[gi][2], gt[gi][3]
        ww, wh = warp[wi][2], warp[wi][3]
        dw = (ww - gw) / gw * 100.0 if gw > 1e-9 else 0.0
        dh = (wh - gh) / gh * 100.0 if gh > 1e-9 else 0.0
        out[wi] = (dw, dh)
    return out


# --------------------------------------------------------------------------------------
# Render sel & panel
# --------------------------------------------------------------------------------------
def _fit_cell(img: np.ndarray) -> np.ndarray:
    """Skala gambar ke lebar CELL_W (jaga rasio)."""
    h, w = img.shape[:2]
    scale = CELL_W / w
    return cv2.resize(img, (CELL_W, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


def _draw_boxes(cell: np.ndarray, boxes_norm: List[Tuple[float, float, float, float]],
                color, width: int = 2) -> None:
    """Gambar box (koordinat ternormalisasi) pada sel yang sudah diskala."""
    ch, cw = cell.shape[:2]
    for b in boxes_norm:
        x1, y1, x2, y2 = norm_to_xyxy(b, cw, ch)
        cv2.rectangle(cell, (int(x1), int(y1)), (int(x2), int(y2)), color, width)


def _black_mask_overlay(cell: np.ndarray) -> np.ndarray:
    """Tandai region hitam (ter-warp keluar frame) dengan tint magenta semi-transparan."""
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    mask = gray < BLACK_THRESH
    if not mask.any():
        return cell
    overlay = cell.copy()
    overlay[mask] = MASK_TINT
    return cv2.addWeighted(overlay, 0.5, cell, 0.5, 0)


def _caption(width: int, lines: List[Tuple[str, tuple, float]]) -> np.ndarray:
    """Bar caption: daftar (teks, warna, skala-font)."""
    bar = np.full((CAP_H, width, 3), CAP_BG, dtype=np.uint8)
    y = 22
    for text, color, scale in lines:
        cv2.putText(bar, text, (10, y), FONT, scale, color, 1, cv2.LINE_AA)
        y += int(20 * (scale / 0.5)) if scale >= 0.5 else 18
    return bar


def _placeholder(h: int) -> np.ndarray:
    ph = np.full((h, CELL_W, 3), (60, 60, 60), dtype=np.uint8)
    cv2.putText(ph, "(gambar asli tidak ada)", (20, h // 2), FONT, 0.6, MUTED, 1, cv2.LINE_AA)
    return ph


def _median(vals: List[float]) -> float:
    return float(np.median(vals)) if vals else 0.0


def compose_panel(ds_id: str, display: str, entry: Dict) -> Optional[np.ndarray]:
    """Bangun panel 3-kolom untuk satu entry. Return BGR uint8 atau None bila rectified gagal."""
    rect = imread(entry["rect_img"])
    if rect is None:
        return None
    rect_cell = _fit_cell(rect)
    ch = rect_cell.shape[0]

    gt = read_yolo_norm(entry["orig_lbl"])
    warp = read_yolo_norm(entry["rect_lbl"])
    pairs = match_gt_warp(gt, warp)
    deltas = size_deltas(gt, warp, pairs)

    # --- Kolom 1: asli + GT hijau ---
    orig = imread(entry["orig_img"]) if entry["orig_img"] else None
    if orig is not None:
        col1 = _fit_cell(orig)
        # samakan tinggi dengan rectified bila beda (harusnya sama dimensi)
        if col1.shape[0] != ch:
            col1 = cv2.resize(col1, (CELL_W, ch), interpolation=cv2.INTER_AREA)
        _draw_boxes(col1, gt, GT_COLOR, 2)
    else:
        col1 = _placeholder(ch)

    # --- Kolom 2: rectified + box warp oranye ---
    col2 = rect_cell.copy()
    _draw_boxes(col2, warp, WARP_COLOR, 2)

    # --- Kolom 3: rectified + mask hitam + box pulih merah + Δ% ---
    col3 = _black_mask_overlay(rect_cell.copy())
    _draw_boxes(col3, warp, RECOVER_COLOR, 2)
    if len(deltas) <= PERBOX_MAX:
        c3h, c3w = col3.shape[:2]
        for wi, (dw, dh) in deltas.items():
            x1, y1, _, _ = norm_to_xyxy(warp[wi], c3w, c3h)
            cv2.putText(col3, f"{dw:+.0f}/{dh:+.0f}", (int(x1), max(10, int(y1) - 3)),
                        FONT, 0.35, WHITE, 1, cv2.LINE_AA)

    dw_med = _median([d[0] for d in deltas.values()])
    dh_med = _median([d[1] for d in deltas.values()])

    # --- caption tiap kolom ---
    cap1 = _caption(CELL_W, [
        (f"{ds_id} | {entry['stem']}", WHITE, 0.5),
        (f"1) ASLI + GT (hijau)  n={len(gt)}", GT_COLOR, 0.5),
    ])
    cap2 = _caption(CELL_W, [
        ("2) RECTIFIED + box warp (oranye)", WHITE, 0.5),
        (f"n_warp={len(warp)}  (drop={max(0, len(gt) - len(warp))})", WARP_COLOR, 0.5),
    ])
    cap3 = _caption(CELL_W, [
        ("3) mask hitam + box pulih (merah)", WHITE, 0.5),
        (f"match={len(pairs)}  d-lebar~{dw_med:+.0f}%  d-tinggi~{dh_med:+.0f}%", RECOVER_COLOR, 0.5),
    ])

    col1 = np.vstack([cap1, col1])
    col2 = np.vstack([cap2, col2])
    col3 = np.vstack([cap3, col3])

    sep = np.full((col1.shape[0], GAP, 3), WHITE, dtype=np.uint8)
    row = np.hstack([col1, sep, col2, sep, col3])

    # header panel
    hdr = np.full((HDR_H, row.shape[1], 3), HDR_BG, dtype=np.uint8)
    cv2.putText(hdr, f"Integritas bbox pasca-MOWA  —  {display}", (12, 30),
                FONT, 0.7, WHITE, 2, cv2.LINE_AA)
    panel = np.vstack([hdr, row])
    cv2.rectangle(panel, (0, 0), (panel.shape[1] - 1, panel.shape[0] - 1), BORDER, 1)
    return panel


# --------------------------------------------------------------------------------------
# Contact-sheet grid
# --------------------------------------------------------------------------------------
def build_grid(display: str, panels: List[np.ndarray], cols: int) -> Optional[np.ndarray]:
    if not panels:
        return None
    thumb_w = 1100
    thumbs = []
    for p in panels:
        s = thumb_w / p.shape[1]
        thumbs.append(cv2.resize(p, (thumb_w, max(1, int(round(p.shape[0] * s)))),
                                 interpolation=cv2.INTER_AREA))
    row_h = max(t.shape[0] for t in thumbs)
    cols = max(1, cols)
    rows = (len(thumbs) + cols - 1) // cols
    pad = 12
    title_h = 52
    grid_w = pad + cols * (thumb_w + pad)
    grid_h = title_h + rows * (row_h + pad) + pad
    canvas = np.full((grid_h, grid_w, 3), WHITE, dtype=np.uint8)
    canvas[:title_h] = HDR_BG
    cv2.putText(canvas, f"Contact sheet integritas bbox — {display}  ({len(thumbs)} panel)",
                (pad, 34), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
    for i, t in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_w + pad)
        y = title_h + r * (row_h + pad) + pad
        canvas[y:y + t.shape[0], x:x + t.shape[1]] = t
    return canvas


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Panel integritas geometri bbox setelah rektifikasi MOWA (tanpa GPU).")
    ap.add_argument("--datasets", nargs="*",
                    default=["pio_val", "broiler_instance_seg", "chicken_detection_fum"])
    ap.add_argument("--limit", type=int, default=12, help="Jumlah gambar per dataset (0 = semua).")
    ap.add_argument("--cols", type=int, default=3, help="Jumlah kolom pada contact-sheet grid.")
    args = ap.parse_args()

    specs = dataset_specs()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    grand_total = 0

    for ds_id in args.datasets:
        if ds_id not in specs:
            print(f"[skip] dataset tak dikenal: {ds_id}")
            continue
        spec = specs[ds_id]
        entries, n_skip = build_pairs(ds_id, spec)
        if not entries:
            print(f"[{ds_id}] tak ada gambar rectified di {RECT_ROOT / ds_id / 'images'} — lewati.")
            continue
        if args.limit > 0:
            entries = entries[:args.limit]

        n_missing_orig = sum(1 for e in entries if e["orig_img"] is None)
        out_dir = OUT_ROOT / ds_id
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{ds_id}] {len(entries)} gambar -> {out_dir}"
              f"  (rectified tanpa asli: {n_skip}; dalam sampel: {n_missing_orig})")

        panels: List[np.ndarray] = []
        n_fail = 0
        for i, e in enumerate(entries, 1):
            panel = compose_panel(ds_id, spec["display"], e)
            if panel is None:
                n_fail += 1
                print(f"  [{i}/{len(entries)}] GAGAL baca rectified: {e['stem']}", file=sys.stderr)
                continue
            imwrite(out_dir / (e["stem"] + ".png"), panel)
            panels.append(panel)

        grid = build_grid(spec["display"], panels, args.cols)
        if grid is not None:
            grid_path = OUT_ROOT / f"{ds_id}_grid.png"
            imwrite(grid_path, grid)
            print(f"  [grid] {grid_path.name}  ({len(panels)} panel, gagal={n_fail})")
        grand_total += len(panels)

    print(f"\n[selesai] {grand_total} panel -> {OUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
