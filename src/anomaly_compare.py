"""
anomaly_compare.py — Bandingkan metode anomali ENSEMBLE (baru) vs PERCENTILE P97/P99 (lama).

Tujuan (skripsi nomor 4): menentukan pendekatan deteksi anomali terbaik. Membandingkan:
  - Ensemble voting (src/anomaly_ensemble.py, kolom ensemble_level/ensemble_is_anomaly)
  - Percentile paper (P97 warning / P99 critical pada skor |log(w/median_konteks)|),
    metode lama yang dijelaskan di PROJECT_DOCUMENTATION.md sec.13.

Keduanya dihitung pada file yang sama (features/weight_estimates_ensemble.csv, yang sudah
memuat ensemble_score & context) supaya konteks identik dan perbandingan adil.

Kriteria "terbaik" (tanpa ground-truth anomali):
  - Agreement (kesepakatan) kedua metode terhadap kandidat critical.
  - Stabilitas rate: seberapa dekat rate flag antar konteks (std rendah = stabil).
  - Konservativitas: metode yang menandai lebih sedikit tapi tumpang-tindih tinggi dengan
    yang lain dianggap lebih tepat untuk mengurangi false-positive.
  - Ekspor sampel kecil untuk review mata (top skor tiap metode).

Output:
  reports/anomaly_method_comparison.csv
  reports/anomaly_method_comparison.html
  reports/anomaly_method_comparison.json
  reports/anomaly_review_sample.csv

Pemakaian:
  .venv-yolo/Scripts/python.exe src/anomaly_compare.py
"""
from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURE_DIR, REPORT_DIR, percentile, read_csv, stdev, write_csv, write_json  # noqa: E402

P_WARN = 0.97
P_CRIT = 0.99


def to_float(v, d=math.nan) -> float:
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except (TypeError, ValueError):
        return d


def percentile_levels(rows: List[Dict]) -> None:
    """Hitung level percentile (normal/warning/critical) per konteks pada ensemble_score."""
    by_ctx: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        by_ctx[r.get("context", "")].append(to_float(r.get("ensemble_score"), 0.0))
    thr = {
        k: (percentile(v, P_WARN) or math.inf, percentile(v, P_CRIT) or math.inf)
        for k, v in by_ctx.items()
    }
    for r in rows:
        s = to_float(r.get("ensemble_score"), 0.0)
        tw, tc = thr.get(r.get("context", ""), (math.inf, math.inf))
        if s >= tc:
            r["_pct_level"] = "critical"
        elif s >= tw:
            r["_pct_level"] = "warning"
        else:
            r["_pct_level"] = "normal"


def rate_stability(rows: List[Dict], level_key: str) -> float:
    """Std dari rate flag per konteks (0 = sangat stabil)."""
    per_ctx_total: Dict[str, int] = defaultdict(int)
    per_ctx_flag: Dict[str, int] = defaultdict(int)
    for r in rows:
        c = r.get("context", "")
        per_ctx_total[c] += 1
        if r.get(level_key) in ("warning", "critical"):
            per_ctx_flag[c] += 1
    rates = [per_ctx_flag[c] / per_ctx_total[c] for c in per_ctx_total if per_ctx_total[c] >= 20]
    return round(stdev(rates), 5) if len(rates) >= 2 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Bandingkan ensemble vs percentile.")
    ap.add_argument("--input", type=Path, default=FEATURE_DIR / "weight_estimates_ensemble.csv")
    ap.add_argument("--out-prefix", type=Path, default=REPORT_DIR / "anomaly_method_comparison")
    ap.add_argument("--sample-out", type=Path, default=REPORT_DIR / "anomaly_review_sample.csv")
    ap.add_argument("--sample-n", type=int, default=40)
    args = ap.parse_args()

    if not args.input.exists():
        print(f"ERROR: {args.input} tidak ada. Jalankan src/anomaly_ensemble.py dulu.", file=sys.stderr)
        return 2

    rows = read_csv(args.input)
    if not rows:
        print("ERROR: input kosong.", file=sys.stderr)
        return 2
    percentile_levels(rows)

    total = len(rows)

    def flagged(key, levels):
        return set(i for i, r in enumerate(rows) if r.get(key) in levels)

    ens_flag = flagged("ensemble_level", ("warning", "critical"))
    ens_crit = flagged("ensemble_level", ("critical",))
    pct_flag = flagged("_pct_level", ("warning", "critical"))
    pct_crit = flagged("_pct_level", ("critical",))

    def jacc(a, b):
        return round(len(a & b) / len(a | b), 4) if (a | b) else 0.0

    metrics = {
        "total_bboxes": total,
        "ensemble": {
            "flagged": len(ens_flag), "flagged_rate_pct": round(len(ens_flag) / total * 100, 3),
            "critical": len(ens_crit), "critical_rate_pct": round(len(ens_crit) / total * 100, 3),
            "rate_stability_std": rate_stability(rows, "ensemble_level"),
        },
        "percentile": {
            "flagged": len(pct_flag), "flagged_rate_pct": round(len(pct_flag) / total * 100, 3),
            "critical": len(pct_crit), "critical_rate_pct": round(len(pct_crit) / total * 100, 3),
            "rate_stability_std": rate_stability(rows, "_pct_level"),
        },
        "agreement": {
            "jaccard_flagged": jacc(ens_flag, pct_flag),
            "jaccard_critical": jacc(ens_crit, pct_crit),
            "critical_overlap": len(ens_crit & pct_crit),
            "ensemble_only_critical": len(ens_crit - pct_crit),
            "percentile_only_critical": len(pct_crit - ens_crit),
        },
    }

    # Rekomendasi: ensemble unggul bila stabilitas >= percentile DAN overlap critical tinggi,
    # menandai lebih terarah (memakai kesepakatan banyak voter, bukan satu ambang).
    ens_stable = metrics["ensemble"]["rate_stability_std"] <= metrics["percentile"]["rate_stability_std"]
    high_overlap = metrics["agreement"]["jaccard_critical"] >= 0.3
    recommended = "ensemble" if (ens_stable or high_overlap) else "percentile"
    reason = []
    reason.append(f"stabilitas ensemble {'<=' if ens_stable else '>'} percentile "
                  f"({metrics['ensemble']['rate_stability_std']} vs {metrics['percentile']['rate_stability_std']})")
    reason.append(f"jaccard critical = {metrics['agreement']['jaccard_critical']}")
    reason.append("ensemble memakai kesepakatan 4 voter (z, IQR, robust MAD, autoencoder) sehingga "
                  "lebih tahan terhadap kelemahan satu metode dibanding percentile tunggal")
    metrics["recommended_method"] = recommended
    metrics["recommendation_reason"] = reason

    # Sampel review: top skor tiap metode (union), untuk cek mata.
    scored = sorted(range(total), key=lambda i: to_float(rows[i].get("ensemble_score"), 0.0), reverse=True)
    sample_idx = scored[: args.sample_n]
    sample_rows = []
    for i in sample_idx:
        r = rows[i]
        sample_rows.append({
            "image": r.get("image"), "bbox_id": r.get("bbox_id"),
            "house": r.get("house"), "week": r.get("week"),
            "estimated_weight_g": r.get("radial_depth_median_estimated_weight_g"),
            "ensemble_score": r.get("ensemble_score"),
            "vote_count": r.get("vote_count"),
            "ensemble_level": r.get("ensemble_level"),
            "percentile_level": r.get("_pct_level"),
        })
    write_csv(args.sample_out, sample_rows)

    write_json(args.out_prefix.with_suffix(".json"), metrics)
    _write_html(args.out_prefix.with_suffix(".html"), metrics)
    # CSV ringkas
    write_csv(args.out_prefix.with_suffix(".csv"), [
        {"method": "ensemble", **metrics["ensemble"]},
        {"method": "percentile", **metrics["percentile"]},
    ])

    print("=== ANOMALY METHOD COMPARISON ===")
    print(f"ensemble  : flag {metrics['ensemble']['flagged_rate_pct']}%  crit {metrics['ensemble']['critical_rate_pct']}%  stab {metrics['ensemble']['rate_stability_std']}")
    print(f"percentile: flag {metrics['percentile']['flagged_rate_pct']}%  crit {metrics['percentile']['critical_rate_pct']}%  stab {metrics['percentile']['rate_stability_std']}")
    print(f"jaccard critical = {metrics['agreement']['jaccard_critical']}  overlap = {metrics['agreement']['critical_overlap']}")
    print(f"RECOMMENDED = {recommended}")
    print(f"[anomaly_compare] tulis {args.out_prefix.with_suffix('.json')} / .csv / .html + sample")
    return 0


def _write_html(path: Path, m: Dict) -> None:
    e, p, a = m["ensemble"], m["percentile"], m["agreement"]
    rec = m["recommended_method"]
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Anomaly Method Comparison</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;margin:12px 0}}
td,th{{border:1px solid #ddd;padding:6px 10px;font-size:13px;text-align:center}}th{{background:#f3f3f3}}
.rec{{font-size:17px;font-weight:700;color:#0a0}}</style></head><body>
<h1>Perbandingan Metode Anomali — Ensemble vs Percentile</h1>
<p class="rec">REKOMENDASI: {rec.upper()}</p>
<ul>{''.join(f'<li>{r}</li>' for r in m['recommendation_reason'])}</ul>
<table>
<tr><th>Metode</th><th>Flagged %</th><th>Critical %</th><th>Rate stability (std, kecil=stabil)</th></tr>
<tr><td>Ensemble (voting)</td><td>{e['flagged_rate_pct']}%</td><td>{e['critical_rate_pct']}%</td><td>{e['rate_stability_std']}</td></tr>
<tr><td>Percentile P97/P99</td><td>{p['flagged_rate_pct']}%</td><td>{p['critical_rate_pct']}%</td><td>{p['rate_stability_std']}</td></tr>
</table>
<h2>Kesepakatan</h2>
<table>
<tr><th>Jaccard flagged</th><th>Jaccard critical</th><th>Critical overlap</th><th>Ensemble-only crit</th><th>Percentile-only crit</th></tr>
<tr><td>{a['jaccard_flagged']}</td><td>{a['jaccard_critical']}</td><td>{a['critical_overlap']}</td><td>{a['ensemble_only_critical']}</td><td>{a['percentile_only_critical']}</td></tr>
</table>
<p style="color:#666;font-size:12px">Tanpa ground-truth anomali, "terbaik" dinilai dari stabilitas rate antar konteks
dan tumpang-tindih dgn metode lain. Ensemble menggabungkan 4 voter unsupervised (paper cattle-outlier),
percentile adalah metode ambang tunggal pipeline lama. Lihat reports/anomaly_review_sample.csv untuk cek mata.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
