from __future__ import annotations

import csv
import html
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "datasets" / "external_datasets.json"
REPORT_DIR = ROOT / "reports" / "external"


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("\n===", " ".join(cmd), "===")
    subprocess.run(cmd, check=True, cwd=ROOT)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    cfg = read_json(CONFIG_PATH)
    relative_datasets = [d for d in cfg.get("datasets", []) if d.get("mode") == "relative"]
    if not relative_datasets:
        raise SystemExit("No relative datasets configured")

    for dataset in relative_datasets:
        did = dataset["dataset_id"]
        run("extract_external_bbox_features.py", "--dataset", did)
        run("relative_anomaly_pipeline.py", "--dataset", did)

    rows = []
    for dataset in relative_datasets:
        did = dataset["dataset_id"]
        summary_path = REPORT_DIR / did / "relative_anomaly_summary.json"
        s = read_json(summary_path)
        camera = s.get("camera_correction", {})
        rows.append(
            {
                "dataset_id": did,
                "display_name": dataset.get("display_name", did),
                "images": s.get("total_images", 0),
                "valid_bbox": s.get("total_bboxes", 0),
                "median_bbox_per_image": s.get("median_bbox_per_image", 0),
                "mean_bbox_per_image": s.get("mean_bbox_per_image", 0),
                "image_cv_median": s.get("image_cv_median", 0),
                "abnormal_image_rate_pct": s.get("abnormal_image_rate_pct", 0),
                "p97_candidate_rate_pct": s.get("candidate_rate_pct", 0),
                "p99_critical_rate_pct": s.get("critical_rate_pct", 0),
                "raw_minor_cv_pct": camera.get("raw_minor_cv_pct", 0),
                "radial_minor_cv_pct": camera.get("radial_minor_cv_pct", 0),
                "radial_depth_minor_cv_pct": camera.get("radial_depth_minor_cv_pct", 0),
                "radial_correction_effect_pct": camera.get("radial_correction_effect_pct", 0),
                "depth_correction_effect_pct": camera.get("depth_correction_effect_pct", 0),
                "percentile_context_counts": json.dumps(s.get("percentile_context_counts", {}), ensure_ascii=False),
                "domain_shift_notes": dataset.get("notes", ""),
            }
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_DIR / "cross_dataset_relative_summary.csv", rows)
    (REPORT_DIR / "cross_dataset_relative_summary.json").write_text(json.dumps({"datasets": rows}, indent=2), encoding="utf-8")
    write_html(rows)
    print("\nDone. Open reports/external/cross_dataset_relative_report.html")


def write_html(rows: List[Dict[str, Any]]) -> None:
    table_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(r['dataset_id']))}</td>"
        f"<td>{r['images']}</td>"
        f"<td>{r['valid_bbox']}</td>"
        f"<td>{r['median_bbox_per_image']}</td>"
        f"<td>{r['image_cv_median']}</td>"
        f"<td>{r['p97_candidate_rate_pct']}%</td>"
        f"<td>{r['p99_critical_rate_pct']}%</td>"
        f"<td>{r['radial_correction_effect_pct']}</td>"
        f"<td>{r['depth_correction_effect_pct']}</td>"
        f"<td>{html.escape(str(r['domain_shift_notes']))}</td>"
        "</tr>"
        for r in rows
    )
    doc = f"""<!doctype html>
<html><head><meta charset=\"utf-8\"><title>Cross-Dataset Relative Generalizability</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%;margin:16px 0}}td,th{{border:1px solid #ddd;padding:6px;font-size:13px;vertical-align:top}}th{{background:#f3f3f3}}code{{background:#f6f6f6;padding:2px 4px}}</style></head>
<body>
<h1>Cross-Dataset Relative Generalizability</h1>
<p><strong>Interpretasi:</strong> semua dataset eksternal dijalankan dalam mode anomaly-relatif. Angka bukan estimasi berat gram aktual, melainkan deviasi ukuran visual terkoreksi terhadap konteks image/dataset.</p>
<table>
<tr><th>Dataset</th><th>Images</th><th>BBox</th><th>Median bbox/img</th><th>Median image CV</th><th>P97+ rate</th><th>P99+ rate</th><th>Radial effect</th><th>Depth effect</th><th>Notes</th></tr>
{table_rows}
</table>
<h2>Kesimpulan awal</h2>
<ul>
<li>P97/P99 sengaja menghasilkan kandidat konservatif; rate mendekati 3% dan 1% pada konteks yang cukup besar.</li>
<li>Dataset sparse harus dibaca sebagai robustness check, bukan pengganti dataset dense commercial-house.</li>
<li>Perbedaan resolusi dan density adalah domain shift utama yang diuji.</li>
</ul>
</body></html>"""
    (REPORT_DIR / "cross_dataset_relative_report.html").write_text(doc, encoding="utf-8")


if __name__ == "__main__":
    main()
