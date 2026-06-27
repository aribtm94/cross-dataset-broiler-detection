from __future__ import annotations

import csv
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ROOT, REPORT_DIR, mean, median, read_csv, stdev, write_csv, write_json  # noqa: E402


DENSE_THRESHOLD = 20
EXTERNAL_REPORT_DIR = REPORT_DIR / "external"


def f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def load_config() -> List[Dict[str, Any]]:
    path = ROOT / "configs" / "datasets" / "external_datasets.json"
    cfg = json.loads(path.read_text(encoding="utf-8"))
    return [d for d in cfg.get("datasets", []) if d.get("mode") == "relative"]


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    did = dataset["dataset_id"]
    report_dir = EXTERNAL_REPORT_DIR / did
    image_rows = read_csv(report_dir / "relative_image_summary.csv")
    enriched_rows = read_csv(report_dir / "relative_enriched_features.csv")

    dense_images = {r["image_relpath"] for r in image_rows if f(r, "count") >= DENSE_THRESHOLD}
    dense_rows = [r for r in enriched_rows if r.get("image_relpath") in dense_images]
    sparse_rows = [r for r in enriched_rows if r.get("image_relpath") not in dense_images]

    def rate(rows: List[Dict[str, Any]], level: str) -> float:
        if not rows:
            return 0.0
        if level == "candidate":
            n = sum(1 for r in rows if r.get("relative_anomaly_level") in {"warning", "critical"})
        else:
            n = sum(1 for r in rows if r.get("relative_anomaly_level") == "critical")
        return round(n / len(rows) * 100, 2)

    resolutions = Counter(f"{int(f(r, 'image_width'))}x{int(f(r, 'image_height'))}" for r in enriched_rows)
    dense_image_counts = [f(r, "count") for r in image_rows if f(r, "count") >= DENSE_THRESHOLD]
    all_image_counts = [f(r, "count") for r in image_rows]
    image_cvs = [f(r, "image_cv_pct") for r in image_rows]
    dense_image_cvs = [f(r, "image_cv_pct") for r in image_rows if f(r, "count") >= DENSE_THRESHOLD]

    # Simple domain-shift indicators.
    distinct_resolutions = len(resolutions)
    top_resolution_rate = round((resolutions.most_common(1)[0][1] / len(enriched_rows) * 100), 2) if enriched_rows and resolutions else 0.0
    density_class = "dense" if (median(all_image_counts) or 0) >= DENSE_THRESHOLD else "sparse_or_mixed"
    resolution_class = "uniform" if distinct_resolutions <= 3 else "mixed_resolution"

    return {
        "dataset_id": did,
        "display_name": dataset.get("display_name", did),
        "total_images": len(image_rows),
        "total_bboxes": len(enriched_rows),
        "dense_threshold": DENSE_THRESHOLD,
        "dense_images": len(dense_images),
        "dense_image_rate_pct": round(len(dense_images) / len(image_rows) * 100, 2) if image_rows else 0.0,
        "dense_bboxes": len(dense_rows),
        "dense_bbox_rate_pct": round(len(dense_rows) / len(enriched_rows) * 100, 2) if enriched_rows else 0.0,
        "sparse_bboxes": len(sparse_rows),
        "all_p97_candidate_rate_pct": rate(enriched_rows, "candidate"),
        "all_p99_critical_rate_pct": rate(enriched_rows, "critical"),
        "dense_p97_candidate_rate_pct": rate(dense_rows, "candidate"),
        "dense_p99_critical_rate_pct": rate(dense_rows, "critical"),
        "sparse_p97_candidate_rate_pct": rate(sparse_rows, "candidate"),
        "sparse_p99_critical_rate_pct": rate(sparse_rows, "critical"),
        "median_bbox_per_image_all": round(median(all_image_counts) or 0.0, 4),
        "median_bbox_per_image_dense": round(median(dense_image_counts) or 0.0, 4),
        "median_image_cv_all": round(median(image_cvs) or 0.0, 4),
        "median_image_cv_dense": round(median(dense_image_cvs) or 0.0, 4),
        "distinct_resolutions": distinct_resolutions,
        "top_resolution_rate_pct": top_resolution_rate,
        "top_resolutions": "; ".join(f"{k}:{v}" for k, v in resolutions.most_common(5)),
        "density_class": density_class,
        "resolution_class": resolution_class,
        "recommended_use": recommended_use(density_class, resolution_class, len(dense_images)),
        "notes": dataset.get("notes", ""),
    }


def recommended_use(density_class: str, resolution_class: str, dense_images: int) -> str:
    if dense_images == 0:
        return "limited_use_sparse_only"
    if density_class == "dense" and resolution_class == "uniform":
        return "strong_generalization_case"
    if density_class == "dense" and resolution_class == "mixed_resolution":
        return "strong_but_domain_shift_case"
    return "domain_shift_or_supporting_case"


def make_html(rows: List[Dict[str, Any]]) -> None:
    tr = "".join(
        "<tr>"
        f"<td>{html.escape(r['dataset_id'])}</td>"
        f"<td>{r['total_images']}</td>"
        f"<td>{r['dense_images']} ({r['dense_image_rate_pct']}%)</td>"
        f"<td>{r['dense_bboxes']} ({r['dense_bbox_rate_pct']}%)</td>"
        f"<td>{r['all_p97_candidate_rate_pct']}%</td>"
        f"<td>{r['dense_p97_candidate_rate_pct']}%</td>"
        f"<td>{r['all_p99_critical_rate_pct']}%</td>"
        f"<td>{r['dense_p99_critical_rate_pct']}%</td>"
        f"<td>{r['distinct_resolutions']}</td>"
        f"<td>{html.escape(r['recommended_use'])}</td>"
        "</tr>"
        for r in rows
    )
    doc = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Dense-only Domain Shift Analysis</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px;vertical-align:top}}th{{background:#f3f3f3}}</style></head>
<body>
<h1>Dense-only Domain Shift Analysis</h1>
<p>Dense image threshold: bbox_count ≥ {DENSE_THRESHOLD}. This analysis separates strong flock/density evidence from sparse robustness cases.</p>
<table>
<tr><th>Dataset</th><th>Images</th><th>Dense images</th><th>Dense bboxes</th><th>All P97+</th><th>Dense P97+</th><th>All P99+</th><th>Dense P99+</th><th>Resolutions</th><th>Recommended use</th></tr>
{tr}
</table>
<h2>Interpretation</h2>
<ul>
<li>Datasets with many dense images are stronger for generalizability claims.</li>
<li>Datasets with many resolutions are domain-shift cases; they are useful, but should be interpreted separately from uniform-resolution datasets.</li>
<li>Sparse datasets are useful to show pipeline robustness, not flock-level anomaly validity.</li>
</ul>
</body></html>"""
    (EXTERNAL_REPORT_DIR / "dense_domain_shift_analysis.html").write_text(doc, encoding="utf-8")


def main() -> None:
    rows = [summarize_dataset(d) for d in load_config()]
    write_csv(EXTERNAL_REPORT_DIR / "dense_domain_shift_analysis.csv", rows)
    write_json(EXTERNAL_REPORT_DIR / "dense_domain_shift_analysis.json", {"dense_threshold": DENSE_THRESHOLD, "datasets": rows})
    make_html(rows)
    for r in rows:
        print(f"{r['dataset_id']}: dense_images={r['dense_images']} dense_p97={r['dense_p97_candidate_rate_pct']} dense_p99={r['dense_p99_critical_rate_pct']} use={r['recommended_use']}")
    print("Wrote reports/external/dense_domain_shift_analysis.html")


if __name__ == "__main__":
    main()
