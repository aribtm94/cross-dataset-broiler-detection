from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DATA_DIR,
    FEATURE_DIR,
    REPORT_DIR,
    ensure_dirs,
    group_by,
    mean,
    median,
    percentile,
    read_csv,
    stdev,
    write_csv,
    write_json,
)


def f(row: Dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except ValueError:
        return default


def estimate(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row.get("house") or "Unknown", int(f(row, "week")))].append(row)

    stats = {}
    for key, items in grouped.items():
        minor_vals = [f(r, "minor_axis") for r in items if f(r, "minor_axis") > 0]
        area_vals = [f(r, "ellipse_area") for r in items if f(r, "ellipse_area") > 0]
        stats[key] = {
            "median_minor_axis": median(minor_vals) or 1.0,
            "median_ellipse_area": median(area_vals) or 1.0,
            "q1_minor_axis": percentile(minor_vals, 0.25) or 0.0,
            "q3_minor_axis": percentile(minor_vals, 0.75) or 0.0,
            "count": len(items),
        }

    output = []
    for row in rows:
        house = row.get("house") or "Unknown"
        week = int(f(row, "week"))
        key = (house, week)
        s = stats[key]
        cobb = f(row, "cobb_weight_g")
        minor = f(row, "minor_axis")
        area = f(row, "ellipse_area")
        minor_ratio = minor / s["median_minor_axis"] if s["median_minor_axis"] else 1.0
        area_ratio = area / s["median_ellipse_area"] if s["median_ellipse_area"] else 1.0

        # Paper favors minor axis + age. Area-based estimate is included as a stabilizing comparison.
        est_minor = cobb * minor_ratio
        est_area = cobb * math.sqrt(area_ratio) if area_ratio > 0 else est_minor
        est_weight = 0.7 * est_minor + 0.3 * est_area
        cobb_diff_g = est_weight - cobb
        cobb_diff_pct = (cobb_diff_g / cobb * 100) if cobb else 0.0

        out = dict(row)
        out.update(
            {
                "group_key": f"{house}_W{week}",
                "group_median_minor_axis": round(s["median_minor_axis"], 4),
                "group_median_ellipse_area": round(s["median_ellipse_area"], 4),
                "minor_ratio_to_group_median": round(minor_ratio, 6),
                "area_ratio_to_group_median": round(area_ratio, 6),
                "estimated_weight_g": round(est_weight, 2),
                "estimated_weight_minor_only_g": round(est_minor, 2),
                "estimated_weight_area_only_g": round(est_area, 2),
                "cobb_diff_g": round(cobb_diff_g, 2),
                "cobb_diff_pct": round(cobb_diff_pct, 2),
            }
        )
        output.append(out)

    # Add within-group z-score after first pass.
    est_groups = defaultdict(list)
    for row in output:
        est_groups[row["group_key"]].append(float(row["estimated_weight_g"]))
    est_stats = {
        k: {
            "mean": mean(v) or 0.0,
            "median": median(v) or 0.0,
            "std": stdev(v),
            "q1": percentile(v, 0.25) or 0.0,
            "q3": percentile(v, 0.75) or 0.0,
        }
        for k, v in est_groups.items()
    }
    for row in output:
        s = est_stats[row["group_key"]]
        est = float(row["estimated_weight_g"])
        std = s["std"]
        z = (est - s["mean"]) / std if std > 0 else 0.0
        iqr = s["q3"] - s["q1"]
        low_fence = s["q1"] - 1.5 * iqr
        high_fence = s["q3"] + 1.5 * iqr
        week_rel = est / s["median"] if s["median"] else 1.0
        cobb_pct = float(row["cobb_diff_pct"])

        flags = []
        if z <= -2 or week_rel < 0.80 or est < low_fence:
            flags.append("below_week_average")
        if z >= 2 or week_rel > 1.20 or est > high_fence:
            flags.append("above_week_average")
        if cobb_pct < -10:
            flags.append("below_cobb_standard")
        if cobb_pct > 10:
            flags.append("above_cobb_standard")
        if cobb_pct < -20 or week_rel < 0.70:
            flags.append("critical_underweight")
        if cobb_pct > 20 or week_rel > 1.30:
            flags.append("critical_overweight")
        if not flags:
            flags.append("normal")

        row.update(
            {
                "group_mean_estimated_weight_g": round(s["mean"], 2),
                "group_median_estimated_weight_g": round(s["median"], 2),
                "group_std_estimated_weight_g": round(s["std"], 2),
                "z_score_group": round(z, 4),
                "relative_to_group_median": round(week_rel, 4),
                "anomaly_flags": "|".join(flags),
                "is_anomaly": "normal" not in flags,
            }
        )
    return output


def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary = []
    for key, items in sorted(group_by(rows, "group_key").items()):
        weights = [float(r["estimated_weight_g"]) for r in items]
        cobb = float(items[0]["cobb_weight_g"] or 0)
        anomaly_items = [r for r in items if str(r["is_anomaly"]) == "True"]
        critical = [r for r in items if "critical" in r["anomaly_flags"]]
        avg = mean(weights) or 0.0
        sd = stdev(weights)
        cv = (sd / avg * 100) if avg else 0.0
        diff_pct = ((avg - cobb) / cobb * 100) if cobb else 0.0
        flags = []
        if diff_pct < -10:
            flags.append("group_below_cobb")
        if diff_pct > 10:
            flags.append("group_above_cobb")
        if cv > 10:
            flags.append("uniformity_problem_cv_gt_10")
        if len(anomaly_items) / len(items) > 0.15:
            flags.append("high_anomaly_rate")
        if not flags:
            flags.append("normal")
        summary.append(
            {
                "group_key": key,
                "house": items[0]["house"],
                "week": items[0]["week"],
                "age_days": items[0]["age_days"],
                "count": len(items),
                "cobb_weight_g": round(cobb, 2),
                "mean_estimated_weight_g": round(avg, 2),
                "median_estimated_weight_g": round(median(weights) or 0.0, 2),
                "std_estimated_weight_g": round(sd, 2),
                "cv_pct": round(cv, 2),
                "mean_cobb_diff_pct": round(diff_pct, 2),
                "anomaly_count": len(anomaly_items),
                "critical_count": len(critical),
                "anomaly_rate_pct": round(len(anomaly_items) / len(items) * 100, 2),
                "group_flags": "|".join(flags),
            }
        )
    return summary


def svg_distribution(summary: List[Dict[str, Any]]) -> str:
    width, height = 900, 420
    pad = 60
    points = []
    for row in summary:
        x = int(row["week"])
        points.append((x, float(row["mean_estimated_weight_g"]), float(row["cobb_weight_g"]), row["group_key"]))
    if not points:
        return "<svg></svg>"
    min_week, max_week = min(p[0] for p in points), max(p[0] for p in points)
    max_y = max(max(p[1], p[2]) for p in points) * 1.1

    def sx(w: int) -> float:
        return pad + (w - min_week) / max(1, max_week - min_week) * (width - 2 * pad)

    def sy(v: float) -> float:
        return height - pad - v / max_y * (height - 2 * pad)

    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#333"/>')
    lines.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#333"/>')
    for w in range(min_week, max_week + 1):
        x = sx(w)
        lines.append(f'<text x="{x}" y="{height-pad+25}" text-anchor="middle" font-size="12">W{w}</text>')
    for frac in [0, .25, .5, .75, 1.0]:
        yv = max_y * frac
        y = sy(yv)
        lines.append(f'<line x1="{pad}" y1="{y}" x2="{width-pad}" y2="{y}" stroke="#ddd"/>')
        lines.append(f'<text x="{pad-8}" y="{y+4}" text-anchor="end" font-size="11">{int(yv)}g</text>')
    cobb_pts = " ".join(f'{sx(p[0])},{sy(p[2])}' for p in sorted(points))
    est_pts = " ".join(f'{sx(p[0])},{sy(p[1])}' for p in sorted(points))
    lines.append(f'<polyline points="{cobb_pts}" fill="none" stroke="#1f77b4" stroke-width="3"/>')
    lines.append(f'<polyline points="{est_pts}" fill="none" stroke="#d62728" stroke-width="3"/>')
    for w, est, cobb, key in points:
        lines.append(f'<circle cx="{sx(w)}" cy="{sy(cobb)}" r="5" fill="#1f77b4"><title>{html.escape(key)} Cobb {cobb:.1f}g</title></circle>')
        lines.append(f'<circle cx="{sx(w)}" cy="{sy(est)}" r="5" fill="#d62728"><title>{html.escape(key)} Est {est:.1f}g</title></circle>')
    lines.append('<text x="680" y="40" font-size="14" fill="#1f77b4">Cobb500 target</text>')
    lines.append('<text x="680" y="62" font-size="14" fill="#d62728">Estimated mean</text>')
    lines.append('<text x="450" y="24" text-anchor="middle" font-size="16" font-weight="bold">Estimated Mean vs Cobb500 Target</text>')
    lines.append('</svg>')
    return "\n".join(lines)


def make_html(summary: List[Dict[str, Any]], top_anomalies: List[Dict[str, Any]]) -> None:
    svg = svg_distribution(summary)
    rows = "\n".join(
        f"<tr><td>{html.escape(str(r['group_key']))}</td><td>{r['count']}</td><td>{r['cobb_weight_g']}</td>"
        f"<td>{r['mean_estimated_weight_g']}</td><td>{r['mean_cobb_diff_pct']}%</td><td>{r['cv_pct']}%</td>"
        f"<td>{r['anomaly_count']}</td><td>{html.escape(str(r['group_flags']))}</td></tr>"
        for r in summary
    )
    anomaly_rows = "\n".join(
        f"<tr><td>{html.escape(str(r['image']))}</td><td>{r['bbox_id']}</td><td>{html.escape(str(r['group_key']))}</td>"
        f"<td>{r['estimated_weight_g']}</td><td>{r['cobb_diff_pct']}%</td><td>{r['z_score_group']}</td>"
        f"<td>{html.escape(str(r['anomaly_flags']))}</td></tr>"
        for r in top_anomalies[:200]
    )
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Broiler Weight Anomaly Report</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>Broiler Weight Anomaly Report</h1>
<p>Metode: estimasi berat relatif dari fitur YOLO <code>minor_axis</code> + umur minggu, dikalibrasi ke target Cobb500 As Hatched.</p>
{svg}
<h2>Ringkasan per grup</h2>
<table><tr><th>Group</th><th>N</th><th>Cobb g</th><th>Mean Est g</th><th>Diff</th><th>CV</th><th>Anomaly</th><th>Flags</th></tr>{rows}</table>
<h2>Top anomalies</h2>
<table><tr><th>Image</th><th>BBox</th><th>Group</th><th>Est g</th><th>Diff Cobb</th><th>Z</th><th>Flags</th></tr>{anomaly_rows}</table>
</body></html>"""
    (REPORT_DIR / "anomaly_report.html").write_text(doc, encoding="utf-8")
    (REPORT_DIR / "plots" / "cobb_vs_estimated.svg").write_text(svg, encoding="utf-8")


def make_overlay_svgs(rows: List[Dict[str, Any]], max_images: int = 12) -> None:
    selected_images = []
    seen = set()
    anomalies = [r for r in rows if str(r["is_anomaly"]) == "True"]
    for r in anomalies:
        key = (r["split"], r["image"])
        if key not in seen:
            selected_images.append(key)
            seen.add(key)
        if len(selected_images) >= max_images:
            break
    by_image = defaultdict(list)
    for r in rows:
        key = (r["split"], r["image"])
        if key in seen:
            by_image[key].append(r)
    for (split, image), items in by_image.items():
        iw = int(float(items[0]["image_width"]))
        ih = int(float(items[0]["image_height"]))
        rel = f"../data/images/{split}/{image}"
        lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{iw}" height="{ih}" viewBox="0 0 {iw} {ih}">']
        lines.append(f'<image href="{html.escape(rel)}" x="0" y="0" width="{iw}" height="{ih}"/>')
        for r in items:
            x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
            flags = str(r["anomaly_flags"])
            color = "#2ca02c"
            if "critical" in flags:
                color = "#d62728"
            elif flags != "normal":
                color = "#ffbf00"
            text = f"{r['estimated_weight_g']}g {r['cobb_diff_pct']}%"
            lines.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="none" stroke="{color}" stroke-width="2"/>')
            lines.append(f'<text x="{x1}" y="{max(12, y1-3)}" font-size="12" fill="{color}" stroke="white" stroke-width="0.4">{html.escape(text)}</text>')
        lines.append("</svg>")
        out = REPORT_DIR / "overlays" / f"{Path(image).stem}.svg"
        out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    feature_path = FEATURE_DIR / "bbox_features.csv"
    rows = [r for r in read_csv(feature_path) if r.get("week") and r.get("cobb_weight_g")]
    estimated = estimate(rows)
    summary = summarize(estimated)

    fieldnames = list(estimated[0].keys()) if estimated else []
    write_csv(FEATURE_DIR / "weight_estimates.csv", estimated, fieldnames)
    anomalies = [r for r in estimated if str(r["is_anomaly"]) == "True"]
    anomalies_sorted = sorted(anomalies, key=lambda r: ("critical" not in r["anomaly_flags"], -abs(float(r["z_score_group"]))))
    write_csv(REPORT_DIR / "anomalies_individual.csv", anomalies_sorted)
    write_csv(REPORT_DIR / "anomalies_by_week.csv", summary)
    write_json(
        REPORT_DIR / "anomaly_summary.json",
        {
            "method": "Cobb500-calibrated relative visual estimate; 70% minor-axis model + 30% sqrt(area) model",
            "total_bboxes": len(estimated),
            "anomaly_count": len(anomalies),
            "anomaly_rate_pct": round(len(anomalies) / len(estimated) * 100, 2) if estimated else 0,
            "groups": summary,
            "cobb_week_assumption": "W1=day7, W2=day14, ..., W6=day42",
        },
    )
    make_html(summary, anomalies_sorted)
    make_overlay_svgs(estimated)

    print(f"Wrote features/weight_estimates.csv ({len(estimated)} rows)")
    print(f"Wrote reports/anomalies_individual.csv ({len(anomalies_sorted)} rows)")
    print("Wrote reports/anomalies_by_week.csv")
    print("Wrote reports/anomaly_report.html")


if __name__ == "__main__":
    main()