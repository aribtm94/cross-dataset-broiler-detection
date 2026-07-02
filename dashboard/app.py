"""
app.py — Dashboard Streamlit sederhana untuk skripsi generalisasi broiler.

Menampilkan, per dataset (PIO val / broiler_instance_seg / chicken_detection_fum):
  - Tab BASELINE  : gambar asli + bounding box label (ground-truth).
  - Tab MOWA      : gambar hasil rectify MOWA + label hasil warp.
  - Tab ANOMALI   : ringkasan metode anomali terpilih + sampel bbox flagged.
  - Panel metrik  : mAP baseline vs MOWA (dari reports/*.json) + verdict A/B.

Dashboard hanya MEMBACA artefak yang sudah dihasilkan pipeline (tidak melatih / tidak
menjalankan MOWA). Deteksi model opsional bisa dijalankan via tombol (butuh .venv-yolo).

Jalankan:
  .venv-yolo/Scripts/python.exe -m streamlit run dashboard/app.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RECT = ROOT / "data" / "rectified"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# (id, display, baseline images dir, baseline labels dir)
DATASETS = [
    ("pio_val", "PIO val (in-domain)",
     ROOT / "data" / "images" / "val", ROOT / "data" / "labels" / "val"),
    ("broiler_instance_seg", "broiler_instance_seg (external)",
     ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "images",
     ROOT / "data" / "external" / "broiler_instance_seg" / "train" / "labels"),
    ("chicken_detection_fum", "chicken_detection_fum (external)",
     ROOT / "data" / "external" / "chicken_detection_fum" / "test" / "images",
     ROOT / "data" / "external" / "chicken_detection_fum" / "test" / "labels"),
]


def load_json(path: Path) -> Optional[Dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def list_images(d: Path) -> List[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def read_boxes(label_path: Path, w: int, h: int) -> List[Tuple[int, int, int, int]]:
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            cx, cy, bw, bh = map(float, parts[1:5])
        except ValueError:
            continue
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        out.append((x1, y1, x2, y2))
    return out


def draw(img: np.ndarray, boxes: List[Tuple[int, int, int, int]], color=(0, 255, 0)) -> np.ndarray:
    out = img.copy()
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    return out


def bgr2rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def metric_row(label: str, base: Optional[float], mowa: Optional[float]) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric(label + " (A)", f"{base:.4f}" if isinstance(base, (int, float)) else "—")
    c2.metric(label + " (B/MOWA)", f"{mowa:.4f}" if isinstance(mowa, (int, float)) else "—")
    if isinstance(base, (int, float)) and isinstance(mowa, (int, float)):
        c3.metric("Δ", f"{mowa - base:+.4f}", delta=f"{mowa - base:+.4f}")
    else:
        c3.metric("Δ", "—")


def find_metric(payload: Optional[Dict], ds_id: str, key: str) -> Optional[float]:
    if not payload:
        return None
    for d in payload.get("datasets", []):
        if d.get("id") == ds_id:
            v = d.get(key)
            return v if isinstance(v, (int, float)) else None
    return None


def main() -> None:
    st.set_page_config(page_title="Broiler Generalizability Dashboard", layout="wide")
    st.title("🐔 Dashboard Generalisasi Broiler — Baseline vs MOWA + Anomali")

    eval_base = load_json(REPORTS / "eval_baseline.json")
    # Utamakan hasil fine-tune (kondisi B') bila ada; jika tidak, MOWA apa adanya (B).
    eval_mowa = load_json(REPORTS / "eval_mowa_ft.json") or load_json(REPORTS / "eval_mowa.json")
    ab = load_json(REPORTS / "ab_comparison_ft.json") or load_json(REPORTS / "ab_comparison.json")
    ens = load_json(REPORTS / "anomaly_ensemble_summary.json")
    ens_cmp = load_json(REPORTS / "anomaly_method_comparison.json")

    # Sidebar
    st.sidebar.header("Pengaturan")
    ds_display = st.sidebar.selectbox("Dataset", [d[1] for d in DATASETS])
    ds = next(d for d in DATASETS if d[1] == ds_display)
    ds_id, _disp, base_img_dir, base_lbl_dir = ds

    imgs = list_images(base_img_dir)
    if not imgs:
        st.warning(f"Tidak ada gambar di {base_img_dir}")
        return
    idx = st.sidebar.slider("Indeks gambar", 0, len(imgs) - 1, 0)
    img_path = imgs[idx]
    st.sidebar.caption(f"{img_path.name}  ({idx + 1}/{len(imgs)})")

    # Verdict banner
    if ab:
        vd = ab.get("verdict", {})
        overall = vd.get("overall", "unknown")
        colors = {"mowa_better": "🟢", "mowa_worse": "🔴", "neutral": "⚪", "unknown": "⚪"}
        st.info(f"{colors.get(overall,'⚪')} **Verdict A/B**: {overall.replace('_',' ').upper()} "
                f"· mean Δ{vd.get('primary_metric','')} = {vd.get('mean_delta_primary')}")

    tab_base, tab_mowa, tab_anom, tab_metrics = st.tabs(
        ["🖼️ Baseline", "🔧 MOWA rectified", "⚠️ Anomali", "📊 Metrik"])

    with tab_base:
        img = cv2.imread(str(img_path))
        if img is None:
            st.error("Gagal baca gambar.")
        else:
            h, w = img.shape[:2]
            boxes = read_boxes(base_lbl_dir / f"{img_path.stem}.txt", w, h)
            st.caption(f"{len(boxes)} bbox label (ground-truth) · {w}×{h}")
            st.image(bgr2rgb(draw(img, boxes)), use_container_width=True)

    with tab_mowa:
        rect_img_dir = RECT / ds_id / "images"
        rect_lbl_dir = RECT / ds_id / "labels"
        rect_img = rect_img_dir / f"{img_path.stem}.jpg"
        if not rect_img.exists():
            # coba ekstensi lain
            cands = list(rect_img_dir.glob(f"{img_path.stem}.*")) if rect_img_dir.is_dir() else []
            rect_img = cands[0] if cands else rect_img
        if rect_img.exists():
            rimg = cv2.imread(str(rect_img))
            h, w = rimg.shape[:2]
            rboxes = read_boxes(rect_lbl_dir / f"{img_path.stem}.txt", w, h)
            st.caption(f"MOWA rectified · {len(rboxes)} bbox warp · {w}×{h}")
            st.image(bgr2rgb(draw(rimg, rboxes, color=(0, 200, 255))), use_container_width=True)
        else:
            st.warning(f"Belum ada hasil MOWA untuk dataset ini di {rect_img_dir}. "
                       f"Jalankan mowa_rectify.py --label-mode warp.")

    with tab_anom:
        if ens_cmp:
            rec = ens_cmp.get("recommended_method", "?")
            st.subheader(f"Metode terpilih: {rec.upper()}")
            c1, c2 = st.columns(2)
            e = ens_cmp.get("ensemble", {})
            p = ens_cmp.get("percentile", {})
            c1.markdown(f"**Ensemble voting**\n\n- flag: {e.get('flagged_rate_pct')}%\n"
                        f"- critical: {e.get('critical_rate_pct')}%\n- stabilitas: {e.get('rate_stability_std')}")
            c2.markdown(f"**Percentile P97/P99**\n\n- flag: {p.get('flagged_rate_pct')}%\n"
                        f"- critical: {p.get('critical_rate_pct')}%\n- stabilitas: {p.get('rate_stability_std')}")
            ag = ens_cmp.get("agreement", {})
            st.caption(f"Jaccard critical = {ag.get('jaccard_critical')} · overlap = {ag.get('critical_overlap')} bbox")
        if ens:
            st.markdown("**Ringkasan ensemble (semua dataset PIO):**")
            st.json({k: ens[k] for k in ("total_bboxes", "flagged_bboxes", "flagged_rate_pct",
                                         "critical_bboxes", "critical_rate_pct", "voters") if k in ens})
        sample = REPORTS / "anomaly_review_sample.csv"
        if sample.exists():
            import csv as _csv
            with sample.open(encoding="utf-8") as fh:
                rows = list(_csv.DictReader(fh))
            st.markdown("**Sampel bbox skor tertinggi (untuk cek mata):**")
            st.dataframe(rows[:40], use_container_width=True)
        if not (ens or ens_cmp):
            st.warning("Belum ada hasil anomali. Jalankan src/anomaly_ensemble.py & src/anomaly_compare.py.")

    with tab_metrics:
        st.subheader(f"Metrik deteksi — {ds_display}")
        for key, lbl in [("map50", "mAP50"), ("map50_95", "mAP50-95"),
                         ("precision", "Precision"), ("recall", "Recall")]:
            metric_row(lbl, find_metric(eval_base, ds_id, key), find_metric(eval_mowa, ds_id, key))
        st.divider()
        st.caption("A = baseline (gambar asli), B = MOWA rectified + label warp. "
                   "Dataset external berbeda domain dari data latih PIO.")
        if not eval_base:
            st.warning("reports/eval_baseline.json belum ada. Jalankan src/eval_detection.py.")


if __name__ == "__main__":
    main()
