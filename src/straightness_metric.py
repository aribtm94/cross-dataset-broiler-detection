"""
straightness_metric.py — Metrik KELURUSAN GARIS (line-segment straightness) untuk
evaluasi objektif rektifikasi fisheye MOWA.

Tujuan
------
Menilai secara objektif apakah MOWA (dan varian iteratif) benar-benar MELURUSKAN citra —
serta apakah ia OVER-/UNDER-correction. Prinsipnya klasik "plumb-line": garis yang lurus di
dunia nyata (rel, garis feeder, tepi kandang/pen) HARUS lurus pula setelah rektifikasi.
Jadi citra yang terektifikasi baik akan (a) menghasilkan segmen garis lurus yang lebih
panjang & lebih banyak, dan (b) menyisakan "bowing" (lengkungan) yang lebih kecil pada
kontur tepi struktural. Kalau MOWA under-correct, garis masih membusur; kalau over-correct,
garis membusur ke arah berlawanan — keduanya menghasilkan residual > 0. Nilai residual
paling kecil = paling lurus.

Referensi
---------
- Xue, Z. et al. "Learning to Calibrate Straight Lines for Fisheye Image Rectification",
  CVPR 2019 — memakai garis-lurus (straight-line/plumb-line) sebagai supervisi kalibrasi.
- Klasik plumb-line + LSD: `cv2.createLineSegmentDetector` (von Gioi et al., LSD).

Metrik (didefinisikan presisi — ini metrik riset, kejelasan > kecanggihan)
-------------------------------------------------------------------------
Per citra dihitung TIGA kolom supaya skripsi bebas memilih:

1. `n_segments`      — jumlah segmen garis lurus (LSD, atau Hough sebagai fallback) yang
                       panjangnya >= `MIN_SEG_FRAC` * diagonal citra. Segmen LSD/Hough sudah
                       lurus SECARA KONSTRUKSI, jadi angka mentahnya bukan residual; tapi pada
                       adegan struktural, citra yang lurus cenderung memberi segmen lebih banyak.
2. `mean_segment_len`— rata-rata panjang segmen tersebut, DINORMALISASI oleh diagonal citra
                       (fraksi 0..1) supaya bisa dibandingkan lintas resolusi (MOWA bisa
                       meng-crop FOV / mengubah ukuran). Lebih besar = garis lurus lebih panjang.
3. `curvature_residual` — SKALAR KELURUSAN UTAMA. "Mean absolute deviation of edge points
                       from a fitted straight line", diagregasi per citra. Implementasi:
                       Canny -> findContours (kontur tepi rapat) -> ambil kontur yang cukup
                       panjang & LINE-LIKE (elongasi tinggi via PCA) -> untuk tiap kontur, garis
                       lurus yang di-fit = sumbu utama PCA (total least squares); residual =
                       rata-rata |jarak tegak-lurus titik ke sumbu|. Diagregasi (rata-rata
                       berbobot panjang busur) lalu dinormalisasi oleh diagonal dan dikali 1000
                       -> satuan "piksel simpangan per 1000 px diagonal". Garis lurus -> ~0;
                       garis membusur (fisheye under/over-correct) -> > 0. LEBIH KECIL = LEBIH LURUS.

Catatan detektor
----------------
- `cv2.createLineSegmentDetector` bisa raise di sebagian build OpenCV (impl/patent dilepas).
  Dibungkus try/except; kalau gagal, fallback ke `cv2.HoughLinesP` dan detektor yang dipakai
  dicatat. `curvature_residual` TIDAK bergantung pada detektor segmen (pakai Canny+contours),
  jadi tetap dihitung meski detektor segmen tak tersedia.
- Kalau OpenCV sama sekali tak ada -> tulis JSON "opencv unavailable" dan exit 0 (graceful).
- Kalau OpenCV ada tapi LSD maupun Hough gagal -> `detector="unavailable"`, kolom segmen kosong,
  tapi `curvature_residual` tetap diisi; exit 0.

Tanpa GPU / tanpa torch. Murni OpenCV + NumPy.

Contoh
------
  .venv-yolo/Scripts/python.exe src/straightness_metric.py \
      --input data/images/val \
      --compare data/rectified/pio_val/images \
      --id pio_val

Output
------
  reports/straightness/<id>_straightness.csv   — baris per citra
  reports/straightness/<id>_summary.json        — rata-rata, detektor, jumlah citra, mean delta
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "straightness"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# --- Parameter metrik (didokumentasikan; dipakai konsisten input vs compare) ---
CANNY_LO, CANNY_HI = 60, 160          # ambang Canny (samakan gaya xue_light_calibration.py)
BLUR_KSIZE = 5                        # GaussianBlur untuk redam noise sebelum Canny
MIN_SEG_FRAC = 0.03                   # panjang minimum segmen (fraksi diagonal) untuk dihitung
MIN_ARC_FRAC = 0.06                   # panjang busur kontur minimum (fraksi diagonal)
MIN_CONTOUR_PTS = 40                  # jumlah titik kontur minimum
MIN_ELONGATION = 0.90                 # 1 - lam2/lam1; hanya kontur "line-like" yang dipakai
RESIDUAL_SCALE = 1000.0               # skala keterbacaan: px simpangan per 1000 px diagonal
HOUGH_THRESHOLD = 120
HOUGH_MAX_GAP = 25


def iter_images(folder: Path, limit: Optional[int] = None) -> List[Path]:
    """Daftar file citra terurut di `folder` (glob ringan, tanpa dependensi berat)."""
    if not folder.is_dir():
        return []
    imgs = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if limit is not None:
        imgs = imgs[:limit]
    return imgs


def _make_segment_detector(cv2) -> Tuple[Optional[Callable], str]:
    """Kembalikan (fungsi_deteksi(gray, edges, diag) -> list panjang segmen, nama_detektor).

    Coba LSD dulu; kalau raise, fallback Hough; kalau dua-duanya gagal -> (None, 'unavailable').
    Fungsi hasil mengembalikan daftar panjang (px) segmen sebelum penyaringan MIN_SEG_FRAC.
    """
    # --- LSD ---
    try:
        lsd = cv2.createLineSegmentDetector()

        def detect_lsd(gray, edges, diag) -> List[float]:
            lines = lsd.detect(gray)[0]
            if lines is None:
                return []
            arr = np.asarray(lines).reshape(-1, 4)
            dx = arr[:, 2] - arr[:, 0]
            dy = arr[:, 3] - arr[:, 1]
            return list(np.hypot(dx, dy))

        # Uji cepat pada citra dummy supaya build yang broken ketahuan sekarang, bukan nanti.
        _ = detect_lsd(np.zeros((8, 8), dtype=np.uint8), None, 10.0)
        return detect_lsd, "LSD"
    except Exception:
        pass

    # --- Hough fallback ---
    try:
        def detect_hough(gray, edges, diag) -> List[float]:
            min_len = int(max(10.0, MIN_SEG_FRAC * diag))
            lines = cv2.HoughLinesP(
                edges, 1, math.pi / 180.0, threshold=HOUGH_THRESHOLD,
                minLineLength=min_len, maxLineGap=HOUGH_MAX_GAP,
            )
            if lines is None:
                return []
            arr = lines.reshape(-1, 4).astype(np.float64)
            dx = arr[:, 2] - arr[:, 0]
            dy = arr[:, 3] - arr[:, 1]
            return list(np.hypot(dx, dy))

        _ = detect_hough(np.zeros((8, 8), dtype=np.uint8), np.zeros((8, 8), dtype=np.uint8), 10.0)
        return detect_hough, "Hough"
    except Exception:
        return None, "unavailable"


def _curvature_residual(cv2, gray: np.ndarray, edges: np.ndarray, diag: float) -> Tuple[Optional[float], int]:
    """Residual kelurusan: rata-rata berbobot |jarak titik kontur ke sumbu-utama PCA|.

    Ambil kontur tepi (Canny) yang panjang & line-like (elongasi tinggi). Untuk tiap kontur,
    'garis lurus yang di-fit' = sumbu utama PCA; residual = mean |proyeksi ke normal|. Lurus -> ~0,
    membusur -> > 0. Dinormalisasi diagonal, dikali RESIDUAL_SCALE. Return (residual, n_kontur_dipakai).
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_arc = MIN_ARC_FRAC * diag
    sum_w = 0.0
    sum_wr = 0.0
    n_used = 0
    for c in contours:
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < MIN_CONTOUR_PTS:
            continue
        arclen = float(cv2.arcLength(c, False))
        if arclen < min_arc:
            continue
        centered = pts - pts.mean(axis=0)
        # PCA via SVD (total least squares): vt[0]=arah utama, vt[1]=normal.
        try:
            _, s, vt = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        lam1 = float(s[0] ** 2)
        lam2 = float(s[1] ** 2) if len(s) > 1 else 0.0
        if lam1 <= 0:
            continue
        elong = 1.0 - lam2 / lam1
        if elong < MIN_ELONGATION:
            continue  # bukan garis (blob/objek melengkung tajam) -> lewati
        normal = vt[1]
        dist = np.abs(centered @ normal)          # jarak tegak-lurus ke sumbu utama
        mad = float(dist.mean())
        sum_wr += mad * arclen
        sum_w += arclen
        n_used += 1
    if sum_w <= 0:
        return None, 0
    residual = (sum_wr / sum_w) / diag * RESIDUAL_SCALE
    return residual, n_used


def compute_metrics(cv2, img_bgr: np.ndarray, detect: Optional[Callable]) -> Dict[str, Any]:
    """Hitung ketiga metrik untuk satu citra BGR."""
    h, w = img_bgr.shape[:2]
    diag = math.hypot(w, h)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_KSIZE, BLUR_KSIZE), 0)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)

    n_segments: Optional[int] = None
    mean_segment_len: Optional[float] = None
    if detect is not None:
        lengths = [L for L in detect(gray, edges, diag) if L >= MIN_SEG_FRAC * diag]
        n_segments = len(lengths)
        mean_segment_len = float(np.mean(lengths) / diag) if lengths else 0.0

    residual, n_contours = _curvature_residual(cv2, gray, edges, diag)
    return {
        "width": w,
        "height": h,
        "n_segments": n_segments,
        "mean_segment_len": mean_segment_len,
        "curvature_residual": residual,
        "n_contours_used": n_contours,
    }


def _round(v: Any, digits: int = 6) -> Any:
    return round(v, digits) if isinstance(v, float) else v


def _mean_opt(values: List[Any]) -> Optional[float]:
    vals = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _round(row.get(k)) for k in fieldnames})


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Metrik kelurusan garis (LSD/Hough + curvature residual) untuk cek over/under-correction MOWA.",
    )
    ap.add_argument("--input", required=True, help="Folder citra (wajib). Mis. data/images/val")
    ap.add_argument("--compare", default=None,
                    help="Folder citra pembanding (opsional), mis. dir rectified; dicocokkan per-stem.")
    ap.add_argument("--id", default=None, help="Label dataset untuk nama file output (default: nama folder --input).")
    ap.add_argument("--limit", type=int, default=None, help="Batasi jumlah citra (debug/smoke).")
    args = ap.parse_args()

    input_dir = Path(args.input)
    compare_dir = Path(args.compare) if args.compare else None
    label = args.id or input_dir.name or "dataset"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{label}_straightness.csv"
    json_path = OUT_DIR / f"{label}_summary.json"

    # --- OpenCV tak tersedia -> graceful exit 0 ---
    try:
        import cv2  # type: ignore
    except Exception as exc:
        report = {
            "id": label,
            "status": "opencv_unavailable",
            "error": str(exc),
            "input_dir": str(input_dir),
            "compare_dir": str(compare_dir) if compare_dir else None,
            "note": "Install opencv-python untuk deteksi garis. Metrik dilewati (exit 0).",
        }
        write_json(json_path, report)
        print(f"[straightness] OpenCV tak tersedia -> tulis {json_path.name} (exit 0)")
        return 0

    detect, detector_name = _make_segment_detector(cv2)
    if detect is None:
        print("[straightness] Detektor segmen (LSD & Hough) tak tersedia; "
              "curvature_residual tetap dihitung, kolom segmen kosong.")

    inputs = iter_images(input_dir, args.limit)
    if not inputs:
        report = {
            "id": label,
            "status": "no_images",
            "input_dir": str(input_dir),
            "detector": detector_name,
            "note": "Folder --input kosong / tidak ada / bukan citra.",
        }
        write_json(json_path, report)
        print(f"[straightness] Tak ada citra di {input_dir} -> tulis {json_path.name} (exit 0)")
        return 0

    compare_map: Dict[str, Path] = {}
    if compare_dir is not None:
        compare_map = {p.stem: p for p in iter_images(compare_dir)}

    rows: List[Dict[str, Any]] = []
    for path in inputs:
        img = cv2.imread(str(path))
        if img is None:
            print(f"[straightness] gagal baca {path.name} — lewati.")
            continue
        m_in = compute_metrics(cv2, img, detect)
        row: Dict[str, Any] = {"stem": path.stem, "detector": detector_name}
        if compare_dir is None:
            row.update({
                "width": m_in["width"],
                "height": m_in["height"],
                "n_segments": m_in["n_segments"],
                "mean_segment_len": m_in["mean_segment_len"],
                "curvature_residual": m_in["curvature_residual"],
                "n_contours_used": m_in["n_contours_used"],
            })
        else:
            cpath = compare_map.get(path.stem)
            m_cmp = None
            if cpath is not None:
                cimg = cv2.imread(str(cpath))
                if cimg is not None:
                    m_cmp = compute_metrics(cv2, cimg, detect)
            row.update({
                "n_segments_input": m_in["n_segments"],
                "n_segments_compare": m_cmp["n_segments"] if m_cmp else None,
                "mean_segment_len_input": m_in["mean_segment_len"],
                "mean_segment_len_compare": m_cmp["mean_segment_len"] if m_cmp else None,
                "curvature_residual_input": m_in["curvature_residual"],
                "curvature_residual_compare": m_cmp["curvature_residual"] if m_cmp else None,
            })
            r_in = m_in["curvature_residual"]
            r_cmp = m_cmp["curvature_residual"] if m_cmp else None
            row["curvature_residual_delta"] = (
                r_cmp - r_in if isinstance(r_in, float) and isinstance(r_cmp, float) else None
            )
        rows.append(row)

    # --- Tulis CSV ---
    if compare_dir is None:
        fieldnames = ["stem", "detector", "width", "height",
                      "n_segments", "mean_segment_len", "curvature_residual", "n_contours_used"]
    else:
        fieldnames = ["stem", "detector",
                      "n_segments_input", "n_segments_compare",
                      "mean_segment_len_input", "mean_segment_len_compare",
                      "curvature_residual_input", "curvature_residual_compare",
                      "curvature_residual_delta"]
    write_csv(csv_path, rows, fieldnames)

    # --- Ringkasan JSON ---
    summary: Dict[str, Any] = {
        "id": label,
        "status": "completed",
        "detector": detector_name,
        "input_dir": str(input_dir),
        "compare_dir": str(compare_dir) if compare_dir else None,
        "n_images": len(rows),
        "metric_notes": (
            "curvature_residual = mean |jarak titik tepi ke sumbu-utama PCA| kontur line-like, "
            "dinormalisasi diagonal x1000 (px/1000px). LEBIH KECIL = LEBIH LURUS. "
            "mean_segment_len = fraksi diagonal. n_segments = jumlah segmen >= "
            f"{MIN_SEG_FRAC} x diagonal ({detector_name})."
        ),
        "params": {
            "canny": [CANNY_LO, CANNY_HI], "blur_ksize": BLUR_KSIZE,
            "min_seg_frac": MIN_SEG_FRAC, "min_arc_frac": MIN_ARC_FRAC,
            "min_contour_pts": MIN_CONTOUR_PTS, "min_elongation": MIN_ELONGATION,
            "residual_scale": RESIDUAL_SCALE,
        },
    }
    if compare_dir is None:
        summary["input"] = {
            "mean_n_segments": _mean_opt([r.get("n_segments") for r in rows]),
            "mean_segment_len": _mean_opt([r.get("mean_segment_len") for r in rows]),
            "mean_curvature_residual": _mean_opt([r.get("curvature_residual") for r in rows]),
        }
    else:
        summary["input"] = {
            "mean_n_segments": _mean_opt([r.get("n_segments_input") for r in rows]),
            "mean_segment_len": _mean_opt([r.get("mean_segment_len_input") for r in rows]),
            "mean_curvature_residual": _mean_opt([r.get("curvature_residual_input") for r in rows]),
        }
        summary["compare"] = {
            "mean_n_segments": _mean_opt([r.get("n_segments_compare") for r in rows]),
            "mean_segment_len": _mean_opt([r.get("mean_segment_len_compare") for r in rows]),
            "mean_curvature_residual": _mean_opt([r.get("curvature_residual_compare") for r in rows]),
            "n_matched": sum(1 for r in rows if isinstance(r.get("curvature_residual_compare"), float)),
        }
        summary["mean_curvature_residual_delta"] = _mean_opt(
            [r.get("curvature_residual_delta") for r in rows]
        )
        summary["interpretation"] = (
            "delta = compare - input pada curvature_residual. delta < 0 -> compare LEBIH LURUS "
            "(garis lebih lurus setelah rektifikasi). delta > 0 -> compare kurang lurus "
            "(indikasi over-/under-correction MOWA)."
        )
    write_json(json_path, summary)

    delta_note = ""
    if compare_dir is not None and isinstance(summary.get("mean_curvature_residual_delta"), float):
        delta_note = f"  mean dResidual={summary['mean_curvature_residual_delta']:.3f}"
    print(f"[straightness] id={label} detector={detector_name} n={len(rows)}{delta_note}")
    print(f"[straightness] tulis {csv_path}")
    print(f"[straightness] tulis {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
