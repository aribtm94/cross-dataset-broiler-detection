from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import FEATURE_DIR, REPORT_DIR, mean, median, percentile, read_csv, stdev, write_csv, write_json  # noqa: E402


BIN_COUNT = 6
MIN_IMAGE_PERCENTILE_COUNT = 20
WEIGHT_PROXY_KEY = "radial_depth_corrected_minor_axis"


def f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def bin_id(v: float, n: int = BIN_COUNT) -> int:
    return max(0, min(n - 1, int(v * n)))


def clamp(v: float, lo: float = 0.55, hi: float = 1.85) -> float:
    if not math.isfinite(v) or v <= 0:
        return 1.0
    return max(lo, min(hi, v))


def mad(values: Iterable[float], center: float) -> float:
    return median(abs(v - center) for v in values) or 0.0


def group_rows(rows: Iterable[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return dict(grouped)


def image_key(row: Dict[str, Any]) -> str:
    return f"{row.get('source_split')}::{row.get('image_relpath')}"


def fallback_context_key(row: Dict[str, Any]) -> str:
    return f"dataset::{row.get('dataset_id')}::{row.get('source_split')}"


def add_camera_corrections(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    raw_median = median(f(r, "minor_axis") for r in rows if f(r, "minor_axis") > 0) or 1.0

    raw_vals = [f(r, "minor_axis") for r in rows if f(r, "minor_axis") > 0]
    raw_cv = (stdev(raw_vals) / (mean(raw_vals) or 1.0) * 100) if raw_vals else 0.0

    radial_bins: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        ratio = f(row, "minor_axis") / raw_median if raw_median else 1.0
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

    radial_vals = [f(r, "radial_corrected_minor_axis") for r in rows if f(r, "radial_corrected_minor_axis") > 0]
    radial_cv = (stdev(radial_vals) / (mean(radial_vals) or 1.0) * 100) if radial_vals else 0.0
    radial_median = median(radial_vals) or 1.0

    y_bins: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        ratio = f(row, "radial_corrected_minor_axis") / radial_median if radial_median else 1.0
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

    depth_vals = [f(r, "radial_depth_corrected_minor_axis") for r in rows if f(r, "radial_depth_corrected_minor_axis") > 0]
    depth_cv = (stdev(depth_vals) / (mean(depth_vals) or 1.0) * 100) if depth_vals else 0.0

    return {
        "method": "Relative mode: radial correction by radius_norm and depth-light correction by bottom_y_norm. Values remain visual-size proxies, not grams.",
        "bin_count": BIN_COUNT,
        "radial_factors_by_bin": radial_factors,
        "perspective_factors_by_bottom_y_bin": perspective_factors,
        "raw_minor_cv_pct": round(raw_cv, 4),
        "radial_minor_cv_pct": round(radial_cv, 4),
        "radial_depth_minor_cv_pct": round(depth_cv, 4),
        "radial_correction_effect_pct": round(raw_cv - radial_cv, 4),
        "depth_correction_effect_pct": round(radial_cv - depth_cv, 4),
    }


def image_quality_flags(count: int, cv: float) -> str:
    flags = []
    if count < 5:
        flags.append("very_low_sample_count")
    elif count < MIN_IMAGE_PERCENTILE_COUNT:
        flags.append("low_sample_count")
    if cv > 30:
        flags.append("critical_uniformity_problem_cv_gt_30")
    elif cv > 20:
        flags.append("warning_uniformity_problem_cv_gt_20")
    if not flags:
        flags.append("normal_image")
    return "|".join(flags)


def percentile_level(score: float, p97: float, p99: float) -> str:
    # Zero-variance contexts (common in one-bbox images) produce P97=P99=0.
    # A zero score means the object equals its context median and should not be
    # flagged just because score >= 0.
    if score <= 0:
        return "normal"
    if p99 > 0 and score >= p99:
        return "critical"
    if p97 > 0 and score >= p97:
        return "warning"
    return "normal"


def run_dataset(dataset_id: str) -> Dict[str, Any]:
    feature_dir = FEATURE_DIR / "external" / dataset_id
    report_dir = REPORT_DIR / "external" / dataset_id
    rows = read_csv(feature_dir / "bbox_features.csv")
    rows = [r for r in rows if f(r, "minor_axis") > 0]
    if not rows:
        raise SystemExit(f"No bbox rows for {dataset_id}. Run extract_external_bbox_features.py first.")

    correction_report = add_camera_corrections(rows)

    by_image = group_rows(rows, image_key)
    image_summary: List[Dict[str, Any]] = []
    image_stats: Dict[str, Dict[str, Any]] = {}
    enriched: List[Dict[str, Any]] = []

    for key, items in sorted(by_image.items()):
        vals = [f(r, WEIGHT_PROXY_KEY) for r in items if f(r, WEIGHT_PROXY_KEY) > 0]
        avg = mean(vals) or 0.0
        med = median(vals) or 0.0
        sd = stdev(vals)
        cv = (sd / avg * 100) if avg else 0.0
        mdev = mad(vals, med)
        flags = image_quality_flags(len(items), cv)
        stat = {
            "dataset_id": dataset_id,
            "source_split": items[0].get("source_split"),
            "image": items[0].get("image"),
            "image_relpath": items[0].get("image_relpath"),
            "count": len(items),
            "image_mean_corrected_minor_axis": round(avg, 4),
            "image_median_corrected_minor_axis": round(med, 4),
            "image_std_corrected_minor_axis": round(sd, 4),
            "image_mad_corrected_minor_axis": round(mdev, 4),
            "image_cv_pct": round(cv, 4),
            "image_flags": flags,
            "image_is_abnormal": flags != "normal_image",
        }
        image_summary.append(stat)
        image_stats[key] = stat

    for row in rows:
        s = image_stats[image_key(row)]
        x = f(row, WEIGHT_PROXY_KEY)
        med = float(s["image_median_corrected_minor_axis"])
        mdev = float(s["image_mad_corrected_minor_axis"])
        rel = x / med if med else 1.0
        rz = 0.6745 * (x - med) / mdev if mdev > 0 else 0.0
        score = abs(math.log(rel)) if rel > 0 else 0.0
        context_key = image_key(row) if int(s["count"]) >= MIN_IMAGE_PERCENTILE_COUNT else fallback_context_key(row)
        out = dict(row)
        out.update(
            {
                "image_mean_corrected_minor_axis": s["image_mean_corrected_minor_axis"],
                "image_median_corrected_minor_axis": s["image_median_corrected_minor_axis"],
                "image_mad_corrected_minor_axis": s["image_mad_corrected_minor_axis"],
                "image_cv_pct": s["image_cv_pct"],
                "relative_to_image_median": round(rel, 6),
                "robust_z_image": round(rz, 6),
                "relative_percentile_score": round(score, 8),
                "relative_percentile_context": context_key,
                "relative_percentile_context_type": "image" if int(s["count"]) >= MIN_IMAGE_PERCENTILE_COUNT else "dataset_split",
                "image_flags": s["image_flags"],
            }
        )
        enriched.append(out)

    score_groups = group_rows(enriched, lambda r: str(r["relative_percentile_context"]))
    thresholds: Dict[str, Dict[str, float]] = {}
    for key, items in score_groups.items():
        scores = [f(r, "relative_percentile_score") for r in items]
        thresholds[key] = {
            "p97": percentile(scores, 0.97) or 0.0,
            "p99": percentile(scores, 0.99) or 0.0,
            "count": float(len(items)),
        }

    for row in enriched:
        t = thresholds[str(row["relative_percentile_context"])]
        score = f(row, "relative_percentile_score")
        level = percentile_level(score, t["p97"], t["p99"])
        row.update(
            {
                "relative_percentile_threshold_p97": round(t["p97"], 8),
                "relative_percentile_threshold_p99": round(t["p99"], 8),
                "relative_percentile_context_count": int(t["count"]),
                "relative_anomaly_level": level,
                "relative_anomaly_flags": "relative_p99_critical" if level == "critical" else "relative_p97_warning" if level == "warning" else "normal",
            }
        )

    candidates = [r for r in enriched if r["relative_anomaly_level"] in {"warning", "critical"}]
    critical = [r for r in enriched if r["relative_anomaly_level"] == "critical"]
    abnormal_images = [r for r in image_summary if str(r["image_is_abnormal"]) == "True"]
    context_counts = defaultdict(int)
    for row in enriched:
        context_counts[str(row["relative_percentile_context_type"])] += 1

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "relative_image_summary.csv", image_summary)
    write_csv(report_dir / "relative_individual_anomalies.csv", candidates)
    write_csv(report_dir / "relative_critical_anomalies.csv", critical)
    write_csv(report_dir / "relative_enriched_features.csv", enriched)

    image_cvs = [f(r, "image_cv_pct") for r in image_summary]
    bbox_counts = [f(r, "count") for r in image_summary]
    summary = {
        "dataset_id": dataset_id,
        "mode": "relative",
        "interpretation_warning": "Visual relative-size anomaly only. No Cobb500, no age metadata, no actual gram-weight claim.",
        "total_images": len(image_summary),
        "total_bboxes": len(enriched),
        "median_bbox_per_image": round(median(bbox_counts) or 0.0, 4),
        "mean_bbox_per_image": round(mean(bbox_counts) or 0.0, 4),
        "abnormal_images": len(abnormal_images),
        "abnormal_image_rate_pct": round(len(abnormal_images) / len(image_summary) * 100, 2) if image_summary else 0.0,
        "image_cv_median": round(median(image_cvs) or 0.0, 4),
        "candidate_bboxes_p97_plus": len(candidates),
        "candidate_rate_pct": round(len(candidates) / len(enriched) * 100, 2) if enriched else 0.0,
        "critical_bboxes_p99_plus": len(critical),
        "critical_rate_pct": round(len(critical) / len(enriched) * 100, 2) if enriched else 0.0,
        "percentile_method": "T = perc(k, X), k=97 warning and k=99 critical. X = abs(log(radial_depth_corrected_minor_axis / image_median_corrected_minor_axis)).",
        "percentile_context_counts": dict(context_counts),
        "min_image_percentile_count": MIN_IMAGE_PERCENTILE_COUNT,
        "camera_correction": correction_report,
        "limitations": [
            "Relative mode is not a calibrated weight estimator.",
            "Sparse datasets may fall back to dataset/split-level percentile context.",
            "Mixed-resolution datasets are valid but may show stronger domain shift.",
        ],
    }
    write_json(report_dir / "relative_anomaly_summary.json", summary)
    make_html(dataset_id, summary, image_summary, candidates, critical, report_dir)
    print(f"{dataset_id}: images={len(image_summary)} bboxes={len(enriched)} p97+={len(candidates)} p99+={len(critical)}")
    return summary


def make_html(dataset_id: str, summary: Dict[str, Any], image_summary: List[Dict[str, Any]], candidates: List[Dict[str, Any]], critical: List[Dict[str, Any]], report_dir: Path) -> None:
    rows = "".join(
        f"<tr><td>{html.escape(str(r['image_relpath']))}</td><td>{r['count']}</td><td>{r['image_cv_pct']}</td><td>{html.escape(str(r['image_flags']))}</td></tr>"
        for r in sorted(image_summary, key=lambda x: f(x, "image_cv_pct"), reverse=True)[:100]
    )
    cand_rows = "".join(
        f"<tr><td>{html.escape(str(r['image_relpath']))}</td><td>{r['bbox_id']}</td><td>{r['relative_to_image_median']}</td><td>{r['robust_z_image']}</td><td>{r['relative_percentile_score']}</td><td>{r['relative_anomaly_level']}</td></tr>"
        for r in candidates[:200]
    )
    doc = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>{html.escape(dataset_id)} Relative Anomaly Report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>{html.escape(dataset_id)} — Relative Anomaly Report</h1>
<p><strong>Important:</strong> visual relative-size anomaly only. No Cobb500 and no actual gram-weight claim.</p>
<ul>
<li>Total images: {summary['total_images']}</li>
<li>Total bboxes: {summary['total_bboxes']}</li>
<li>P97+ candidates: {summary['candidate_bboxes_p97_plus']} ({summary['candidate_rate_pct']}%)</li>
<li>P99+ critical: {summary['critical_bboxes_p99_plus']} ({summary['critical_rate_pct']}%)</li>
<li>Median bbox/image: {summary['median_bbox_per_image']}</li>
<li>Median image CV: {summary['image_cv_median']}</li>
</ul>
<h2>Top image CV / sample-count flags</h2>
<table><tr><th>Image</th><th>Count</th><th>CV %</th><th>Flags</th></tr>{rows}</table>
<h2>Relative anomaly candidates (first 200)</h2>
<table><tr><th>Image</th><th>BBox</th><th>Relative</th><th>Robust Z</th><th>Score</th><th>Level</th></tr>{cand_rows}</table>
</body></html>"""
    (report_dir / "relative_anomaly_report.html").write_text(doc, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()
    run_dataset(args.dataset)


if __name__ == "__main__":
    main()
