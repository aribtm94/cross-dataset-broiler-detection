"""
anomaly_ensemble.py — Deteksi anomali berat broiler dengan VOTING ENSEMBLE unsupervised.

Mengadaptasi paper:
  "Multi-algorithmic approach for detecting outliers in cattle intake data"
  (J. Agriculture & Food Research, 2024; configs/1-s2.0-S2666154324000589-main.pdf)

Paper itu menggabungkan beberapa detektor outlier unsupervised lalu MEM-VOTING hasilnya
(sebuah titik dianggap outlier bila mayoritas model setuju). Karena data kita adalah bbox
berat-relatif per gambar (bukan deret waktu intake), voter "time-series decomposition"
diganti voter statistik lain yang setara-secara-semangat. Empat voter:

  1. z-score global (per konteks)         : |z| >= Z_THRESH
  2. IQR fence (Tukey)                     : x < Q1-1.5*IQR atau x > Q3+1.5*IQR
  3. robust MAD z-score                    : |0.6745*(x-med)/MAD| >= ROBUST_THRESH
  4. autoencoder rekonstruksi (jika torch) : reconstruction error di ekor atas
     (fallback: percentile P97/P99 ala pipeline lama bila torch tak ada)

Skor anomali satu-arah (ayam terlalu kecil ATAU terlalu besar), konsisten dgn pipeline lama:
  score = |log(estimated_weight / context_median_weight)|

Konteks (context) mengikuti pipeline lama: per-image bila bbox_count_image >= 100,
selain itu per house-week (kolom group_key).

Input : features/weight_estimates_compare.csv  (kolom radial_depth_median_estimated_weight_g)
Output:
  features/weight_estimates_ensemble.csv        (+ kolom voter & vote_count & ensemble_level)
  reports/anomaly_ensemble_summary.json
  reports/anomaly_ensemble_report.html

Pemakaian:
  .venv-yolo/Scripts/python.exe src/anomaly_ensemble.py
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

# Impor util pipeline (common.py ada di src/, sebelah file ini).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_DIR, REPORT_DIR, median, percentile, read_csv, write_csv, write_json  # noqa: E402

WEIGHT_KEY = "radial_depth_median_estimated_weight_g"
MIN_IMAGE_COUNT = 100  # >= ini -> konteks per image; selain itu per house-week
Z_THRESH = 2.5
ROBUST_THRESH = 3.5
AE_TAIL_P = 0.97  # ekor atas reconstruction error dianggap anomali (voter AE)
VOTE_MAJORITY = 2  # minimal setuju agar ditandai (dari 4 voter)


def to_float(v, default=math.nan) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def context_key(row: Dict[str, str]) -> str:
    """Kunci konteks: per-image bila gambar padat, else per house-week (group_key)."""
    return row["_ctx"]


def assign_contexts(rows: List[Dict]) -> None:
    """Isi row['_ctx']: image bila image punya >= MIN_IMAGE_COUNT bbox, else group_key."""
    by_image: Dict[str, int] = defaultdict(int)
    for r in rows:
        by_image[r.get("image", "")] += 1
    for r in rows:
        img = r.get("image", "")
        if by_image[img] >= MIN_IMAGE_COUNT:
            r["_ctx"] = f"img::{img}"
        else:
            gk = f"{r.get('house')}_W{r.get('week')}"
            r["_ctx"] = f"grp::{gk}"


def score_rows(rows: List[Dict]) -> None:
    """Hitung skor satu-arah |log(w/context_median)| dan simpan di row['_score']."""
    groups: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        w = to_float(r.get(WEIGHT_KEY))
        r["_w"] = w
        if math.isfinite(w) and w > 0:
            groups[context_key(r)].append(w)
    ctx_median = {k: (median(v) or 0.0) for k, v in groups.items()}
    for r in rows:
        w = r["_w"]
        med = ctx_median.get(context_key(r), 0.0)
        if med > 0 and math.isfinite(w) and w > 0:
            r["_score"] = abs(math.log(w / med))
        else:
            r["_score"] = 0.0


def _mad(vals: List[float], med: float) -> float:
    if not vals:
        return 0.0
    dev = [abs(v - med) for v in vals]
    return median(dev) or 0.0


def voter_statistics(rows: List[Dict]) -> None:
    """Voter 1-3: z-score, IQR fence, robust MAD — dihitung per konteks pada _w."""
    ctx: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        if math.isfinite(r["_w"]) and r["_w"] > 0:
            ctx[context_key(r)].append(r["_w"])

    stats = {}
    for k, vals in ctx.items():
        n = len(vals)
        mean_v = sum(vals) / n if n else 0.0
        var = sum((v - mean_v) ** 2 for v in vals) / (n - 1) if n >= 2 else 0.0
        std = math.sqrt(var)
        med = median(vals) or 0.0
        q1 = percentile(vals, 0.25) or 0.0
        q3 = percentile(vals, 0.75) or 0.0
        iqr = q3 - q1
        stats[k] = {
            "mean": mean_v, "std": std, "median": med,
            "low_fence": q1 - 1.5 * iqr, "high_fence": q3 + 1.5 * iqr,
            "mad": _mad(vals, med),
        }

    for r in rows:
        s = stats.get(context_key(r))
        w = r["_w"]
        if not s or not (math.isfinite(w) and w > 0):
            r["v_zscore"] = r["v_iqr"] = r["v_robust"] = 0
            continue
        z = (w - s["mean"]) / s["std"] if s["std"] > 0 else 0.0
        r["v_zscore"] = int(abs(z) >= Z_THRESH)
        r["v_iqr"] = int(w < s["low_fence"] or w > s["high_fence"])
        rz = 0.6745 * (w - s["median"]) / s["mad"] if s["mad"] > 0 else 0.0
        r["v_robust"] = int(abs(rz) >= ROBUST_THRESH)


def voter_autoencoder(rows: List[Dict]) -> str:
    """Voter 4: autoencoder kecil pada fitur bbox ternormalisasi-konteks.

    Melatih AE dense ringan (unsupervised) memakai torch bila tersedia; titik dgn
    reconstruction error di ekor atas (per konteks) ditandai. Bila torch tak ada,
    fallback ke percentile P97 pada _score (voter setara pipeline lama).
    Return nama metode voter yang dipakai.
    """
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except Exception:
        _voter_percentile_fallback(rows)
        return "percentile_fallback"

    # Fitur per bbox: berat, skor, dan beberapa fitur geometri bila ada.
    feat_keys = [WEIGHT_KEY, "minor_axis", "ellipse_area", "radius_norm", "bottom_y_norm"]
    valid = [r for r in rows if math.isfinite(r["_w"]) and r["_w"] > 0]
    if len(valid) < 200:
        _voter_percentile_fallback(rows)
        return "percentile_fallback_small_n"

    X = []
    for r in valid:
        X.append([to_float(r.get(k), 0.0) for k in feat_keys])
    X = np.asarray(X, dtype=np.float32)
    # Standarisasi kolom.
    mu = X.mean(0, keepdims=True)
    sd = X.std(0, keepdims=True)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd

    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(Xn).to(dev)
    d = Xn.shape[1]
    ae = nn.Sequential(
        nn.Linear(d, 8), nn.ReLU(), nn.Linear(8, 3), nn.ReLU(),
        nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, d),
    ).to(dev)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-2)
    lossf = nn.MSELoss()
    ae.train()
    for _ in range(60):
        opt.zero_grad()
        out = ae(xt)
        loss = lossf(out, xt)
        loss.backward()
        opt.step()
    ae.eval()
    with torch.no_grad():
        err = ((ae(xt) - xt) ** 2).mean(1).cpu().numpy()

    # Ambang ekor atas per konteks.
    per_ctx: Dict[str, List[float]] = defaultdict(list)
    for r, e in zip(valid, err):
        per_ctx[context_key(r)].append(float(e))
    thr = {k: (percentile(v, AE_TAIL_P) or math.inf) for k, v in per_ctx.items()}

    for r in rows:
        r["v_ae"] = 0
    for r, e in zip(valid, err):
        r["v_ae"] = int(float(e) >= thr.get(context_key(r), math.inf))
    return f"autoencoder({dev.type})"


def _voter_percentile_fallback(rows: List[Dict]) -> None:
    """Fallback voter 4: P97 pada _score per konteks (setara metode pipeline lama)."""
    per_ctx: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        per_ctx[context_key(r)].append(r["_score"])
    thr = {k: (percentile(v, 0.97) or math.inf) for k, v in per_ctx.items()}
    for r in rows:
        r["v_ae"] = int(r["_score"] >= thr.get(context_key(r), math.inf))


def combine_votes(rows: List[Dict]) -> None:
    for r in rows:
        votes = int(r.get("v_zscore", 0)) + int(r.get("v_iqr", 0)) + \
                int(r.get("v_robust", 0)) + int(r.get("v_ae", 0))
        r["vote_count"] = votes
        if votes >= 3:
            r["ensemble_level"] = "critical"
        elif votes >= VOTE_MAJORITY:
            r["ensemble_level"] = "warning"
        else:
            r["ensemble_level"] = "normal"
        r["ensemble_is_anomaly"] = int(votes >= VOTE_MAJORITY)


def main() -> int:
    ap = argparse.ArgumentParser(description="Voting ensemble anomaly detection (unsupervised).")
    ap.add_argument("--input", type=Path, default=FEATURE_DIR / "weight_estimates_compare.csv")
    ap.add_argument("--out-csv", type=Path, default=FEATURE_DIR / "weight_estimates_ensemble.csv")
    ap.add_argument("--out-json", type=Path, default=REPORT_DIR / "anomaly_ensemble_summary.json")
    ap.add_argument("--out-html", type=Path, default=REPORT_DIR / "anomaly_ensemble_report.html")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} tidak ada. Jalankan pipeline fitur dulu "
              f"(extract_bbox_features + estimate + compare_camera_corrections).", file=sys.stderr)
        return 2

    rows = [r for r in read_csv(args.input) if r.get("week") and to_float(r.get(WEIGHT_KEY)) > 0]
    if not rows:
        print("ERROR: tidak ada baris valid (butuh week + berat).", file=sys.stderr)
        return 2

    assign_contexts(rows)
    score_rows(rows)
    voter_statistics(rows)
    ae_method = voter_autoencoder(rows)
    combine_votes(rows)

    total = len(rows)
    n_warn = sum(1 for r in rows if r["ensemble_level"] == "warning")
    n_crit = sum(1 for r in rows if r["ensemble_level"] == "critical")
    n_flag = n_warn + n_crit
    voter_rates = {
        v: round(sum(int(r.get(v, 0)) for r in rows) / total * 100, 3)
        for v in ["v_zscore", "v_iqr", "v_robust", "v_ae"]
    }

    # Tulis CSV (buang kolom internal _ prefix kecuali skor yang berguna).
    out_rows = []
    for r in rows:
        rr = {k: v for k, v in r.items() if not k.startswith("_")}
        rr["ensemble_score"] = round(r["_score"], 6)
        rr["context"] = r["_ctx"]
        out_rows.append(rr)
    write_csv(args.out_csv, out_rows)

    summary = {
        "method": "unsupervised voting ensemble (cattle-outlier paper, adapted)",
        "voters": ["z_score", "iqr_fence", "robust_mad", ae_method],
        "vote_majority": VOTE_MAJORITY,
        "weight_key": WEIGHT_KEY,
        "context_rule": f"per-image if bbox>= {MIN_IMAGE_COUNT} else per house-week",
        "total_bboxes": total,
        "warning_bboxes": n_warn,
        "critical_bboxes": n_crit,
        "flagged_bboxes": n_flag,
        "flagged_rate_pct": round(n_flag / total * 100, 3),
        "critical_rate_pct": round(n_crit / total * 100, 3),
        "voter_flag_rate_pct": voter_rates,
    }
    write_json(args.out_json, summary)
    _write_html(args.out_html, summary)

    print(f"[ensemble] total={total} flagged={n_flag} ({summary['flagged_rate_pct']}%) "
          f"critical={n_crit} ({summary['critical_rate_pct']}%)")
    print(f"[ensemble] voter rates: {voter_rates}  AE={ae_method}")
    print(f"[ensemble] tulis {args.out_csv} / {args.out_json} / {args.out_html}")
    return 0


def _write_html(path: Path, s: Dict) -> None:
    vr = s["voter_flag_rate_pct"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Anomaly Voting Ensemble</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ddd;padding:6px 10px;font-size:13px}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:1px 4px}}</style></head><body>
<h1>Deteksi Anomali — Voting Ensemble (unsupervised)</h1>
<p>Metode: {s['method']}. Voter: <code>{', '.join(map(str, s['voters']))}</code>.
Ditandai bila &ge; {s['vote_majority']} voter setuju. Konteks: {s['context_rule']}.</p>
<table>
<tr><th>Total bbox</th><td>{s['total_bboxes']}</td></tr>
<tr><th>Flagged (warning+critical)</th><td>{s['flagged_bboxes']} ({s['flagged_rate_pct']}%)</td></tr>
<tr><th>Critical (&ge;3 voter)</th><td>{s['critical_bboxes']} ({s['critical_rate_pct']}%)</td></tr>
</table>
<h2>Tingkat penandaan per voter</h2>
<table><tr><th>z-score</th><th>IQR</th><th>robust MAD</th><th>AE/percentile</th></tr>
<tr><td>{vr['v_zscore']}%</td><td>{vr['v_iqr']}%</td><td>{vr['v_robust']}%</td><td>{vr['v_ae']}%</td></tr></table>
<p style="color:#666;font-size:12px">Anomali di sini adalah outlier relatif dalam konteks gambar/house-week,
bukan diagnosa medis. Voting mengurangi false-positive dari satu metode tunggal.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
