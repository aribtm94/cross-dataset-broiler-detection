from __future__ import annotations

import html
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

from common import FEATURE_DIR, REPORT_DIR, ensure_dirs, mean, median, percentile, read_csv, stdev, write_csv, write_json


BIN_COUNT = 6
MODEL_SPECS = [
    ("original_median", "minor_axis", "ellipse_area", "median"),
    ("original_mean", "minor_axis", "ellipse_area", "mean"),
    ("radial_median", "radial_corrected_minor_axis", "radial_corrected_ellipse_area", "median"),
    ("radial_mean", "radial_corrected_minor_axis", "radial_corrected_ellipse_area", "mean"),
    ("radial_depth_median", "radial_depth_corrected_minor_axis", "radial_depth_corrected_ellipse_area", "median"),
    ("radial_depth_mean", "radial_depth_corrected_minor_axis", "radial_depth_corrected_ellipse_area", "mean"),
]


def f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def group_key(row: Dict[str, Any]) -> str:
    return f"{row.get('house')}_W{row.get('week')}"


def bin_id(v: float, n: int = BIN_COUNT) -> int:
    return max(0, min(n - 1, int(v * n)))


def clamp(v: float, lo: float = 0.55, hi: float = 1.85) -> float:
    if not math.isfinite(v) or v <= 0:
        return 1.0
    return max(lo, min(hi, v))


def grouped(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[group_key(row)].append(row)
    return dict(out)


def add_camera_corrections(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups = grouped(rows)
    raw_group_medians = {
        key: median(f(r, "minor_axis") for r in items if f(r, "minor_axis") > 0) or 1.0 for key, items in groups.items()
    }

    radial_bins: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        base = raw_group_medians[group_key(row)]
        ratio = f(row, "minor_axis") / base if base else 1.0
        radial_bins[bin_id(f(row, "radius_norm"))].append(ratio)
    radial_factors = {b: clamp(median(vals) or 1.0) for b, vals in radial_bins.items()}
    for b in range(BIN_COUNT):
        radial_factors.setdefault(b, 1.0)

    for row in rows:
        rf = radial_factors[bin_id(f(row, "radius_norm"))]
        row["radial_bin"] = bin_id(f(row, "radius_norm"))
        row["radial_scale_factor"] = round(rf, 6)
        row["radial_corrected_minor_axis"] = round(f(row, "minor_axis") / rf, 4)
        row["radial_corrected_ellipse_area"] = round(f(row, "ellipse_area") / (rf * rf), 4)

    radial_group_medians = {
        key: median(f(r, "radial_corrected_minor_axis") for r in items if f(r, "radial_corrected_minor_axis") > 0) or 1.0
        for key, items in groups.items()
    }
    y_bins: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        base = radial_group_medians[group_key(row)]
        ratio = f(row, "radial_corrected_minor_axis") / base if base else 1.0
        y_bins[bin_id(f(row, "bottom_y_norm"))].append(ratio)
    perspective_factors = {b: clamp(median(vals) or 1.0) for b, vals in y_bins.items()}
    for b in range(BIN_COUNT):
        perspective_factors.setdefault(b, 1.0)

    for row in rows:
        pf = perspective_factors[bin_id(f(row, "bottom_y_norm"))]
        row["perspective_bin"] = bin_id(f(row, "bottom_y_norm"))
        row["perspective_scale_factor"] = round(pf, 6)
        row["radial_depth_corrected_minor_axis"] = round(f(row, "radial_corrected_minor_axis") / pf, 4)
        row["radial_depth_corrected_ellipse_area"] = round(f(row, "radial_corrected_ellipse_area") / (pf * pf), 4)

    return {
        "method": "DaFIR-light radial correction from radius_norm + depth-light perspective correction from bottom_y_norm.",
        "radial_factors_by_bin": radial_factors,
        "perspective_factors_by_bottom_y_bin": perspective_factors,
        "bin_count": BIN_COUNT,
        "notes": [
            "Radial factor uses median bbox-size ratio per radius bin. It approximates fisheye/radial bias, not full DaFIR flow-map rectification.",
            "Perspective factor uses median bbox-size ratio per bottom_y bin. It approximates depth/perspective bias, not learned monocular depth reprojection.",
        ],
    }


def baseline(items: List[Dict[str, Any]], value_key: str, mode: str) -> float:
    vals = [f(r, value_key) for r in items if f(r, value_key) > 0]
    if mode == "mean":
        return mean(vals) or 1.0
    return median(vals) or 1.0


def flag_estimate(est: float, cobb: float, stats: Dict[str, float]) -> Tuple[str, bool, float, float]:
    std = stats["std"]
    z = (est - stats["mean"]) / std if std > 0 else 0.0
    rel_median = est / stats["median"] if stats["median"] else 1.0
    rel_mean = est / stats["mean"] if stats["mean"] else 1.0
    iqr = stats["q3"] - stats["q1"]
    low_fence = stats["q1"] - 1.5 * iqr
    high_fence = stats["q3"] + 1.5 * iqr
    cobb_pct = ((est - cobb) / cobb * 100) if cobb else 0.0
    flags = []
    if z <= -2 or rel_mean < 0.80 or rel_median < 0.80 or est < low_fence:
        flags.append("below_week_average")
    if z >= 2 or rel_mean > 1.20 or rel_median > 1.20 or est > high_fence:
        flags.append("above_week_average")
    if cobb_pct < -10:
        flags.append("below_cobb_standard")
    if cobb_pct > 10:
        flags.append("above_cobb_standard")
    if cobb_pct < -20 or rel_mean < 0.70 or rel_median < 0.70:
        flags.append("critical_underweight")
    if cobb_pct > 20 or rel_mean > 1.30 or rel_median > 1.30:
        flags.append("critical_overweight")
    if not flags:
        flags.append("normal")
    return "|".join(flags), "normal" not in flags, z, cobb_pct


def add_model_estimates(rows: List[Dict[str, Any]]) -> None:
    groups = grouped(rows)
    bases: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for gkey, items in groups.items():
        for model, minor_key, area_key, mode in MODEL_SPECS:
            bases[(gkey, model)] = (baseline(items, minor_key, mode), baseline(items, area_key, mode))

    for row in rows:
        cobb = f(row, "cobb_weight_g")
        for model, minor_key, area_key, _mode in MODEL_SPECS:
            base_minor, base_area = bases[(group_key(row), model)]
            minor_ratio = f(row, minor_key) / base_minor if base_minor else 1.0
            area_ratio = f(row, area_key) / base_area if base_area else 1.0
            est_minor = cobb * minor_ratio
            est_area = cobb * math.sqrt(area_ratio) if area_ratio > 0 else est_minor
            est = 0.7 * est_minor + 0.3 * est_area
            row[f"{model}_estimated_weight_g"] = round(est, 2)
            row[f"{model}_minor_ratio"] = round(minor_ratio, 6)
            row[f"{model}_area_ratio"] = round(area_ratio, 6)

    model_stats: Dict[Tuple[str, str], Dict[str, float]] = {}
    for gkey, items in groups.items():
        for model, *_ in MODEL_SPECS:
            vals = [f(r, f"{model}_estimated_weight_g") for r in items]
            model_stats[(gkey, model)] = {
                "mean": mean(vals) or 0.0,
                "median": median(vals) or 0.0,
                "std": stdev(vals),
                "q1": percentile(vals, 0.25) or 0.0,
                "q3": percentile(vals, 0.75) or 0.0,
            }

    for row in rows:
        cobb = f(row, "cobb_weight_g")
        anomaly_models = []
        normal_models = []
        for model, *_ in MODEL_SPECS:
            est = f(row, f"{model}_estimated_weight_g")
            flags, is_anom, z, cobb_pct = flag_estimate(est, cobb, model_stats[(group_key(row), model)])
            row[f"{model}_flags"] = flags
            row[f"{model}_is_anomaly"] = is_anom
            row[f"{model}_z_score"] = round(z, 4)
            row[f"{model}_cobb_diff_pct"] = round(cobb_pct, 2)
            if is_anom:
                anomaly_models.append(model)
            else:
                normal_models.append(model)
        row["anomaly_model_count"] = len(anomaly_models)
        row["normal_model_count"] = len(normal_models)
        row["anomaly_models"] = "|".join(anomaly_models) if anomaly_models else "none"
        row["consensus_status"] = (
            "all_models_anomaly"
            if len(anomaly_models) == len(MODEL_SPECS)
            else "all_models_normal"
            if not anomaly_models
            else "mixed"
        )


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    groups = grouped(rows)
    for model, *_ in MODEL_SPECS:
        for gkey, items in sorted(groups.items()):
            vals = [f(r, f"{model}_estimated_weight_g") for r in items]
            anomalies = [r for r in items if str(r[f"{model}_is_anomaly"]) == "True"]
            critical = [r for r in anomalies if "critical" in str(r[f"{model}_flags"])]
            avg = mean(vals) or 0.0
            sd = stdev(vals)
            cobb = f(items[0], "cobb_weight_g")
            out.append(
                {
                    "model": model,
                    "group_key": gkey,
                    "house": items[0].get("house"),
                    "week": items[0].get("week"),
                    "count": len(items),
                    "cobb_weight_g": round(cobb, 2),
                    "mean_estimated_weight_g": round(avg, 2),
                    "median_estimated_weight_g": round(median(vals) or 0.0, 2),
                    "std_estimated_weight_g": round(sd, 2),
                    "cv_pct": round((sd / avg * 100) if avg else 0.0, 2),
                    "mean_cobb_diff_pct": round(((avg - cobb) / cobb * 100) if cobb else 0.0, 2),
                    "anomaly_count": len(anomalies),
                    "critical_count": len(critical),
                    "anomaly_rate_pct": round(len(anomalies) / len(items) * 100, 2),
                }
            )
    return out


def write_html(summary: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> None:
    model_totals = []
    for model, *_ in MODEL_SPECS:
        anomalies = [r for r in rows if str(r[f"{model}_is_anomaly"]) == "True"]
        model_totals.append((model, len(anomalies), len(anomalies) / len(rows) * 100 if rows else 0.0))
    model_rows = "".join(
        f"<tr><td>{html.escape(model)}</td><td>{cnt}</td><td>{rate:.2f}%</td></tr>" for model, cnt, rate in model_totals
    )
    summary_rows = "".join(
        f"<tr><td>{html.escape(str(r['model']))}</td><td>{html.escape(str(r['group_key']))}</td><td>{r['count']}</td>"
        f"<td>{r['mean_estimated_weight_g']}</td><td>{r['cv_pct']}%</td><td>{r['mean_cobb_diff_pct']}%</td>"
        f"<td>{r['anomaly_count']}</td><td>{r['anomaly_rate_pct']}%</td></tr>"
        for r in summary
    )
    consensus = defaultdict(int)
    for row in rows:
        consensus[row["consensus_status"]] += 1
    consensus_rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{v}</td></tr>" for k, v in sorted(consensus.items()))
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Camera Correction Baseline Comparison</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>Camera Correction Baseline Comparison</h1>
<p>Membandingkan baseline median/mean, koreksi radial ala DaFIR-light, dan koreksi depth/perspective-light berbasis posisi bawah bbox.</p>
<h2>Total anomaly per model</h2><table><tr><th>Model</th><th>Anomaly count</th><th>Anomaly rate</th></tr>{model_rows}</table>
<h2>Consensus</h2><table><tr><th>Status</th><th>Count</th></tr>{consensus_rows}</table>
<h2>Per model dan grup</h2><table><tr><th>Model</th><th>Group</th><th>N</th><th>Mean g</th><th>CV</th><th>Diff Cobb</th><th>Anomaly</th><th>Rate</th></tr>{summary_rows}</table>
</body></html>"""
    (REPORT_DIR / "anomaly_baseline_comparison.html").write_text(doc, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    rows = [r for r in read_csv(FEATURE_DIR / "bbox_features.csv") if r.get("week") and r.get("cobb_weight_g")]
    correction_report = add_camera_corrections(rows)
    add_model_estimates(rows)
    summary = summarize(rows)
    consensus = [r for r in rows if r["consensus_status"] == "all_models_anomaly"]

    write_csv(FEATURE_DIR / "weight_estimates_compare.csv", rows)
    write_csv(REPORT_DIR / "anomaly_baseline_comparison.csv", summary)
    write_csv(REPORT_DIR / "anomalies_consensus.csv", consensus)
    write_json(REPORT_DIR / "correction_factors.json", correction_report)
    write_json(
        REPORT_DIR / "anomaly_baseline_comparison.json",
        {
            "total_bboxes": len(rows),
            "models": [m[0] for m in MODEL_SPECS],
            "summary": summary,
            "consensus_all_models_anomaly": len(consensus),
            "consensus_all_models_anomaly_rate_pct": round(len(consensus) / len(rows) * 100, 2) if rows else 0.0,
        },
    )
    write_html(summary, rows)
    print(f"Wrote features/weight_estimates_compare.csv ({len(rows)} rows)")
    print("Wrote reports/anomaly_baseline_comparison.csv/html/json")
    print(f"Wrote reports/anomalies_consensus.csv ({len(consensus)} rows)")


if __name__ == "__main__":
    main()