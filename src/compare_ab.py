"""
compare_ab.py — Bandingkan evaluasi deteksi BASELINE (A) vs MOWA-rectified (B).

Membaca dua hasil eval_detection.py:
  reports/eval_baseline.json  (kondisi A, gambar asli)
  reports/eval_mowa.json      (kondisi B, gambar + label hasil MOWA warp)

Menghitung delta per dataset & metrik, lalu memberi VERDICT keseluruhan memakai
metrik primer = rata-rata mAP50-95 lintas dataset (yang punya angka di kedua kondisi).

Output:
  reports/ab_comparison.json
  reports/ab_comparison.csv
  reports/ab_comparison.html

Pemakaian:
  .venv-yolo/Scripts/python.exe src/compare_ab.py
"""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["map50", "map50_95", "precision", "recall"]
PRIMARY = "map50_95"
# Ambang netral: |delta rata-rata| di bawah ini dianggap "setara" (bukan menang/kalah).
NEUTRAL_EPS = 0.005


def load(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Tidak ada {path}. Jalankan eval_detection.py dulu.")
    return json.loads(path.read_text(encoding="utf-8"))


def index_by_id(payload: Dict) -> Dict[str, Dict]:
    return {d["id"]: d for d in payload.get("datasets", [])}


def build_rows(base: Dict, mowa: Dict) -> List[Dict]:
    b_idx, m_idx = index_by_id(base), index_by_id(mowa)
    rows = []
    for ds_id in b_idx:
        b, m = b_idx[ds_id], m_idx.get(ds_id, {})
        row: Dict[str, object] = {
            "dataset": ds_id,
            "display": b.get("display", ds_id),
            "in_domain": b.get("in_domain"),
            "base_status": b.get("status"),
            "mowa_status": m.get("status"),
            "base_images": b.get("images"),
            "mowa_images": m.get("images"),
        }
        for mt in METRICS:
            bv, mv = b.get(mt), m.get(mt)
            row[f"base_{mt}"] = bv
            row[f"mowa_{mt}"] = mv
            row[f"delta_{mt}"] = round(mv - bv, 5) if (isinstance(bv, (int, float)) and isinstance(mv, (int, float))) else None
        rows.append(row)
    return rows


def verdict(rows: List[Dict]) -> Dict:
    deltas = [r[f"delta_{PRIMARY}"] for r in rows if isinstance(r.get(f"delta_{PRIMARY}"), (int, float))]
    per_ds = []
    for r in rows:
        d = r.get(f"delta_{PRIMARY}")
        if isinstance(d, (int, float)):
            label = "better" if d > NEUTRAL_EPS else "worse" if d < -NEUTRAL_EPS else "neutral"
            per_ds.append({"dataset": r["dataset"], "delta_primary": d, "label": label})
    mean_delta = round(sum(deltas) / len(deltas), 5) if deltas else None
    if mean_delta is None:
        overall = "unknown"
    elif mean_delta > NEUTRAL_EPS:
        overall = "mowa_better"
    elif mean_delta < -NEUTRAL_EPS:
        overall = "mowa_worse"
    else:
        overall = "neutral"
    return {
        "primary_metric": PRIMARY,
        "mean_delta_primary": mean_delta,
        "overall": overall,
        "per_dataset": per_ds,
        "n_better": sum(1 for p in per_ds if p["label"] == "better"),
        "n_worse": sum(1 for p in per_ds if p["label"] == "worse"),
        "n_neutral": sum(1 for p in per_ds if p["label"] == "neutral"),
    }


def fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def delta_cell(v: Optional[float]) -> str:
    if not isinstance(v, (int, float)):
        return "<td>—</td>"
    color = "#0a0" if v > NEUTRAL_EPS else "#c00" if v < -NEUTRAL_EPS else "#888"
    sign = "+" if v >= 0 else ""
    return f'<td style="color:{color};font-weight:600">{sign}{v:.4f}</td>'


def write_html(path: Path, rows: List[Dict], vd: Dict, base: Dict, mowa: Dict) -> None:
    trs = []
    for r in rows:
        cells = [
            f'<td>{html.escape(str(r["display"]))}</td>',
            f'<td>{"in-domain" if r["in_domain"] else "external"}</td>',
        ]
        for mt in METRICS:
            cells.append(f"<td>{fmt(r.get(f'base_{mt}'))}</td>")
            cells.append(f"<td>{fmt(r.get(f'mowa_{mt}'))}</td>")
            cells.append(delta_cell(r.get(f"delta_{mt}")))
        trs.append("<tr>" + "".join(cells) + "</tr>")

    metric_headers = "".join(
        f'<th>{m} A</th><th>{m} B</th><th>Δ</th>' for m in METRICS
    )
    verdict_color = {"mowa_better": "#0a0", "mowa_worse": "#c00", "neutral": "#888", "unknown": "#888"}[vd["overall"]]
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>A/B: Baseline vs MOWA</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:6px 8px;font-size:13px;text-align:center}}th{{background:#f3f3f3}}
h1{{margin-bottom:4px}}.verdict{{font-size:18px;font-weight:700;color:{verdict_color}}}</style></head><body>
<h1>Perbandingan A/B — Baseline vs MOWA-rectified</h1>
<p>Model: <code>{html.escape(Path(base.get("weights","?")).name)}</code> ·
A = gambar asli · B = MOWA warp · metrik primer = <b>{PRIMARY}</b></p>
<p class="verdict">VERDICT: {vd["overall"].replace("_"," ").upper()}
(mean Δ {PRIMARY} = {fmt(vd["mean_delta_primary"])}; better {vd["n_better"]} / worse {vd["n_worse"]} / neutral {vd["n_neutral"]})</p>
<table><tr><th>Dataset</th><th>Domain</th>{metric_headers}</tr>
{''.join(trs)}
</table>
<p style="color:#666;font-size:12px;margin-top:12px">A = kondisi baseline, B = kondisi MOWA (gambar rectified + label warp). Δ = B − A;
hijau berarti MOWA lebih baik. Dataset external berbeda domain dari data latih PIO, jadi angka
absolut rendah adalah sinyal generalisasi, bukan bug.</p>
</body></html>"""
    path.write_text(doc, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description="Bandingkan eval baseline vs MOWA.")
    ap.add_argument("--baseline", type=Path, default=ROOT / "reports" / "eval_baseline.json")
    ap.add_argument("--mowa", type=Path, default=ROOT / "reports" / "eval_mowa.json")
    ap.add_argument("--out-prefix", type=Path, default=ROOT / "reports" / "ab_comparison")
    args = ap.parse_args()

    base, mowa = load(args.baseline), load(args.mowa)
    rows = build_rows(base, mowa)
    vd = verdict(rows)

    payload = {
        "baseline_weights": base.get("weights"),
        "mowa_weights": mowa.get("weights"),
        "verdict": vd,
        "rows": rows,
    }
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with args.out_prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        cols = ["dataset", "in_domain"] + [f"{c}_{mt}" for mt in METRICS for c in ("base", "mowa", "delta")]
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["dataset"], r["in_domain"]] +
                       [r.get(f"{c}_{mt}") for mt in METRICS for c in ("base", "mowa", "delta")])

    write_html(args.out_prefix.with_suffix(".html"), rows, vd, base, mowa)

    print("=== A/B VERDICT ===")
    print(f"overall = {vd['overall']}  mean delta {PRIMARY} = {vd['mean_delta_primary']}")
    for p in vd["per_dataset"]:
        print(f"  {p['dataset']:24s} delta {PRIMARY}={p['delta_primary']:+.4f}  [{p['label']}]")
    print(f"[compare_ab] tulis {args.out_prefix.with_suffix('.json')} / .csv / .html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
