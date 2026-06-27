from __future__ import annotations

import html
import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List

from common import FEATURE_DIR, REPORT_DIR, ensure_dirs, mean, median, percentile, read_csv, stdev, write_csv, write_json


WEIGHT_KEY = "radial_depth_median_estimated_weight_g"
MODEL_NAME = "radial_depth_median"
MIN_IMAGE_PERCENTILE_COUNT = 100


def f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def mad(values: Iterable[float], center: float) -> float:
    deviations = [abs(v - center) for v in values]
    return median(deviations) or 0.0


def image_key(row: Dict[str, Any]) -> str:
    return f"{row.get('split')}::{row.get('image')}"


def house_week_key(row: Dict[str, Any]) -> str:
    return f"{row.get('house')}_W{row.get('week')}"


def group_rows(rows: Iterable[Dict[str, Any]], key_fn) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[key_fn(row)].append(row)
    return dict(grouped)


def classify_image(avg: float, cv: float, cobb: float, count: int) -> str:
    diff_pct = ((avg - cobb) / cobb * 100) if cobb else 0.0
    flags = []
    if count < 20:
        flags.append("low_sample_count")
    if diff_pct < -20:
        flags.append("critical_image_below_cobb")
    elif diff_pct < -10:
        flags.append("warning_image_below_cobb")
    if diff_pct > 20:
        flags.append("critical_image_above_cobb")
    elif diff_pct > 10:
        flags.append("warning_image_above_cobb")
    if cv > 30:
        flags.append("critical_uniformity_problem_cv_gt_30")
    elif cv > 20:
        flags.append("warning_uniformity_problem_cv_gt_20")
    if not flags:
        flags.append("normal_image")
    return "|".join(flags)


def classify_individual(rel: float, robust_z: float, model_is_anomaly: bool, image_flags: str) -> str:
    flags = []
    if rel < 0.65 or robust_z < -4.5:
        flags.append("critical_low_vs_image")
    elif rel < 0.75 or robust_z < -3.5:
        flags.append("warning_low_vs_image")
    if rel > 1.35 or robust_z > 4.5:
        flags.append("critical_high_vs_image")
    elif rel > 1.25 or robust_z > 3.5:
        flags.append("warning_high_vs_image")
    if model_is_anomaly:
        flags.append("camera_corrected_model_anomaly")
    if "critical_image" in image_flags:
        flags.append("from_critical_image")
    if not flags:
        flags.append("normal")
    return "|".join(flags)


def severity(flags: str) -> str:
    if "critical" in flags:
        return "critical"
    if "warning" in flags or "camera_corrected_model_anomaly" in flags:
        return "warning"
    return "normal"


def percentile_level(score: float, p97: float, p99: float) -> str:
    if score >= p99:
        return "critical"
    if score >= p97:
        return "warning"
    return "normal"


def main() -> None:
    ensure_dirs()
    rows = [r for r in read_csv(FEATURE_DIR / "weight_estimates_compare.csv") if r.get(WEIGHT_KEY)]
    by_image = group_rows(rows, image_key)

    image_summary: List[Dict[str, Any]] = []
    image_stats: Dict[str, Dict[str, Any]] = {}
    enriched: List[Dict[str, Any]] = []

    for key, items in sorted(by_image.items()):
        weights = [f(r, WEIGHT_KEY) for r in items]
        avg = mean(weights) or 0.0
        med = median(weights) or 0.0
        sd = stdev(weights)
        cv = (sd / avg * 100) if avg else 0.0
        mdev = mad(weights, med)
        cobb = f(items[0], "cobb_weight_g")
        diff_pct = ((avg - cobb) / cobb * 100) if cobb else 0.0
        flags = classify_image(avg, cv, cobb, len(items))
        item = {
            "split": items[0].get("split"),
            "image": items[0].get("image"),
            "house": items[0].get("house"),
            "week": items[0].get("week"),
            "age_days": items[0].get("age_days"),
            "count": len(items),
            "cobb_weight_g": round(cobb, 2),
            "image_mean_weight_g": round(avg, 2),
            "image_median_weight_g": round(med, 2),
            "image_std_weight_g": round(sd, 2),
            "image_mad_weight_g": round(mdev, 2),
            "image_cv_pct": round(cv, 2),
            "image_cobb_diff_pct": round(diff_pct, 2),
            "image_flags": flags,
            "image_is_abnormal": flags != "normal_image",
        }
        image_summary.append(item)
        image_stats[key] = item

    for row in rows:
        s = image_stats[image_key(row)]
        weight = f(row, WEIGHT_KEY)
        med = float(s["image_median_weight_g"])
        mdev = float(s["image_mad_weight_g"])
        rel = weight / med if med else 1.0
        rz = 0.6745 * (weight - med) / mdev if mdev > 0 else 0.0
        model_anomaly = str(row.get(f"{MODEL_NAME}_is_anomaly")) == "True"
        flags = classify_individual(rel, rz, model_anomaly, str(s["image_flags"]))
        percentile_score = abs(math.log(rel)) if rel > 0 else 0.0
        percentile_context = image_key(row) if int(s["count"]) >= MIN_IMAGE_PERCENTILE_COUNT else house_week_key(row)
        out = dict(row)
        out.update(
            {
                "image_mean_weight_g": s["image_mean_weight_g"],
                "image_median_weight_g": s["image_median_weight_g"],
                "image_mad_weight_g": s["image_mad_weight_g"],
                "image_cv_pct": s["image_cv_pct"],
                "image_cobb_diff_pct": s["image_cobb_diff_pct"],
                "relative_to_image_median": round(rel, 4),
                "robust_z_image": round(rz, 4),
                "image_flags": s["image_flags"],
                "image_context_flags": flags,
                "final_anomaly_level": severity(flags),
                "paper_percentile_score": round(percentile_score, 8),
                "paper_percentile_context": percentile_context,
                "paper_percentile_context_type": "image" if int(s["count"]) >= MIN_IMAGE_PERCENTILE_COUNT else "house_week",
            }
        )
        enriched.append(out)

    score_groups = group_rows(enriched, lambda r: str(r["paper_percentile_context"]))
    thresholds: Dict[str, Dict[str, float]] = {}
    for key, items in score_groups.items():
        scores = [f(r, "paper_percentile_score") for r in items]
        thresholds[key] = {
            "p97": percentile(scores, 0.97) or 0.0,
            "p99": percentile(scores, 0.99) or 0.0,
            "count": float(len(items)),
        }

    for row in enriched:
        t = thresholds[str(row["paper_percentile_context"])]
        score = f(row, "paper_percentile_score")
        level = percentile_level(score, t["p97"], t["p99"])
        row.update(
            {
                "paper_percentile_threshold_p97": round(t["p97"], 8),
                "paper_percentile_threshold_p99": round(t["p99"], 8),
                "paper_percentile_context_count": int(t["count"]),
                "paper_percentile_level": level,
                "paper_percentile_flags": "percentile_p99_critical" if level == "critical" else "percentile_p97_warning" if level == "warning" else "normal",
            }
        )

    final_candidates = [r for r in enriched if r["final_anomaly_level"] in {"warning", "critical"}]
    critical = [r for r in enriched if r["final_anomaly_level"] == "critical"]
    abnormal_images = [r for r in image_summary if str(r["image_is_abnormal"]) == "True"]
    percentile_candidates = [r for r in enriched if r["paper_percentile_level"] in {"warning", "critical"}]
    percentile_critical = [r for r in enriched if r["paper_percentile_level"] == "critical"]
    percentile_context_counts = defaultdict(int)
    for row in enriched:
        percentile_context_counts[str(row["paper_percentile_context_type"])] += 1

    write_csv(FEATURE_DIR / "weight_estimates_image_context.csv", enriched)
    write_csv(REPORT_DIR / "image_level_anomalies.csv", image_summary)
    write_csv(REPORT_DIR / "final_individual_anomaly_candidates.csv", final_candidates)
    write_csv(REPORT_DIR / "final_individual_critical_anomalies.csv", critical)
    write_csv(REPORT_DIR / "percentile_paper_individual_anomalies.csv", percentile_candidates)
    write_csv(REPORT_DIR / "percentile_paper_critical_anomalies.csv", percentile_critical)
    write_json(
        REPORT_DIR / "image_level_anomaly_summary.json",
        {
            "model": MODEL_NAME,
            "total_images": len(image_summary),
            "abnormal_images": len(abnormal_images),
            "abnormal_image_rate_pct": round(len(abnormal_images) / len(image_summary) * 100, 2) if image_summary else 0.0,
            "total_bboxes": len(enriched),
            "final_candidate_bboxes": len(final_candidates),
            "final_candidate_rate_pct": round(len(final_candidates) / len(enriched) * 100, 2) if enriched else 0.0,
            "critical_bboxes": len(critical),
            "critical_rate_pct": round(len(critical) / len(enriched) * 100, 2) if enriched else 0.0,
            "paper_percentile_method": "T = perc(k, X), k=97 warning and k=99 critical. X = abs(log(weight / image_median_weight)).",
            "paper_percentile_candidate_bboxes": len(percentile_candidates),
            "paper_percentile_candidate_rate_pct": round(len(percentile_candidates) / len(enriched) * 100, 2) if enriched else 0.0,
            "paper_percentile_critical_bboxes": len(percentile_critical),
            "paper_percentile_critical_rate_pct": round(len(percentile_critical) / len(enriched) * 100, 2) if enriched else 0.0,
            "paper_percentile_context_counts": dict(percentile_context_counts),
            "thresholds": {
                "image_cobb_warning_pct": 10,
                "image_cobb_critical_pct": 20,
                "image_cv_warning_pct": 20,
                "image_cv_critical_pct": 30,
                "individual_warning_relative_to_image_median": "<0.75 or >1.25",
                "individual_critical_relative_to_image_median": "<0.65 or >1.35",
                "individual_warning_robust_z": "abs(z) > 3.5",
                "individual_critical_robust_z": "abs(z) > 4.5",
                "paper_percentile_warning": "score >= P97",
                "paper_percentile_critical": "score >= P99",
                "paper_percentile_image_min_count": MIN_IMAGE_PERCENTILE_COUNT,
            },
        },
    )
    make_html(image_summary, final_candidates, critical)
    make_percentile_html(image_summary, percentile_candidates, percentile_critical, thresholds)
    print(f"Wrote reports/image_level_anomalies.csv ({len(image_summary)} images)")
    print(f"Wrote reports/final_individual_anomaly_candidates.csv ({len(final_candidates)} rows)")
    print(f"Wrote reports/final_individual_critical_anomalies.csv ({len(critical)} rows)")
    print(f"Wrote reports/percentile_paper_individual_anomalies.csv ({len(percentile_candidates)} rows)")
    print(f"Wrote reports/percentile_paper_critical_anomalies.csv ({len(percentile_critical)} rows)")


def make_html(image_summary: List[Dict[str, Any]], candidates: List[Dict[str, Any]], critical: List[Dict[str, Any]]) -> None:
    abnormal_images = [r for r in image_summary if str(r["image_is_abnormal"]) == "True"]
    image_rows = "".join(
        f"<tr><td>{html.escape(str(r['image']))}</td><td>{r['count']}</td><td>{r['cobb_weight_g']}</td>"
        f"<td>{r['image_mean_weight_g']}</td><td>{r['image_cobb_diff_pct']}%</td><td>{r['image_cv_pct']}%</td>"
        f"<td>{html.escape(str(r['image_flags']))}</td></tr>"
        for r in abnormal_images[:300]
    )
    cand_rows = "".join(
        f"<tr><td>{html.escape(str(r['image']))}</td><td>{r['bbox_id']}</td><td>{r[WEIGHT_KEY]}</td>"
        f"<td>{r['relative_to_image_median']}</td><td>{r['robust_z_image']}</td><td>{r['final_anomaly_level']}</td>"
        f"<td>{html.escape(str(r['image_context_flags']))}</td></tr>"
        for r in candidates[:300]
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Image-level Cobb500 Anomaly Report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>Image-level Cobb500 Anomaly Report</h1>
<p>Final report memakai <code>{MODEL_NAME}</code>, rata-rata per image, dan robust z-score/MAD per image.</p>
<p>Total abnormal image: {len(abnormal_images)} / {len(image_summary)}. Candidate bbox: {len(candidates)}. Critical bbox: {len(critical)}.</p>
<h2>Abnormal images</h2><table><tr><th>Image</th><th>N</th><th>Cobb g</th><th>Mean g</th><th>Diff Cobb</th><th>CV</th><th>Flags</th></tr>{image_rows}</table>
<h2>Individual candidates</h2><table><tr><th>Image</th><th>BBox</th><th>Weight g</th><th>Rel image median</th><th>Robust z</th><th>Level</th><th>Flags</th></tr>{cand_rows}</table>
</body></html>"""
    (REPORT_DIR / "image_level_anomaly_report.html").write_text(doc, encoding="utf-8")


def make_percentile_html(
    image_summary: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    critical: List[Dict[str, Any]],
    thresholds: Dict[str, Dict[str, float]],
) -> None:
    cand_rows = "".join(
        f"<tr><td>{html.escape(str(r['image']))}</td><td>{r['bbox_id']}</td><td>{r[WEIGHT_KEY]}</td>"
        f"<td>{r['paper_percentile_score']}</td><td>{r['paper_percentile_threshold_p97']}</td><td>{r['paper_percentile_threshold_p99']}</td>"
        f"<td>{html.escape(str(r['paper_percentile_context_type']))}</td><td>{r['paper_percentile_level']}</td></tr>"
        for r in candidates[:300]
    )
    threshold_rows = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td>{int(v['count'])}</td><td>{v['p97']:.6f}</td><td>{v['p99']:.6f}</td></tr>"
        for k, v in list(thresholds.items())[:300]
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Paper Percentile Anomaly Report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>Paper Percentile Anomaly Report</h1>
<p>Threshold mengikuti percentile method: <code>T = perc(k, X)</code>, dengan <code>k=97</code> untuk warning dan <code>k=99</code> untuk critical.</p>
<p>Anomaly score: <code>abs(log(radial_depth_median_estimated_weight_g / image_median_weight_g))</code>.</p>
<p>Candidate P97+: {len(candidates)}. Critical P99+: {len(critical)}. Total image: {len(image_summary)}.</p>
<h2>Percentile candidates</h2><table><tr><th>Image</th><th>BBox</th><th>Weight g</th><th>Score</th><th>P97</th><th>P99</th><th>Context</th><th>Level</th></tr>{cand_rows}</table>
<h2>Threshold contexts</h2><table><tr><th>Context</th><th>N</th><th>P97</th><th>P99</th></tr>{threshold_rows}</table>
</body></html>"""
    (REPORT_DIR / "percentile_paper_anomaly_report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()