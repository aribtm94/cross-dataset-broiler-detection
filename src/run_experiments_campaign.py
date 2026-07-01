"""
run_experiments_campaign.py — Orkestrator Task 4: re-test seluruh varian MOWA pada 3 dataset.

Menyusun SATU tabel master hasil eksperimen (variants x datasets) dengan Δ mAP50-95
terhadap baseline + verdict per varian. Script ini HANYA merangkai evaluasi (chaining)
dan agregasi; ia mengimpor ulang logika evaluasi dari `eval_detection.py` (tidak
mengimplementasi ulang) dan ambang verdict dari `compare_ab.py`.

Varian yang dinilai (lihat docs/EXPERIMENT_PLAN_MOWA_V2.md untuk konteks metodologi):
  - baseline        : bobot baseline, gambar asli (referensi Δ).
  - mowa_1pass      : bobot baseline, gambar hasil MOWA 1 pass (data/rectified).
  - mowa_1pass_ft   : bobot fine-tune-on-rectified, gambar MOWA 1 pass.
  - mowa_iter2      : bobot baseline, gambar MOWA 2 pass iteratif (data/rectified_iter2).
  - enhanced        : bobot baseline, gambar CLAHE+unsharp (data/enhanced).
  - tta             : hasil eksternal dari eval_detection_tta.py (di-merge dari JSON).
  - radial_retrain  : bobot hasil retrain dengan augmentasi distorsi radial, gambar asli.

Setiap varian yang input-nya belum tersedia TIDAK membuat script crash — ia dicatat
dengan status (missing_weights / missing_input / external / no_data) supaya tabel jujur
soal apa yang benar-benar dijalankan.

Metrik primer verdict = rata-rata Δ mAP50-95 lintas dataset (dead-band NEUTRAL_EPS=0.005).

Jalankan di bawah .venv-yolo (evaluate_one memakai ultralytics; GPU hanya dipakai saat
benar-benar mengevaluasi). Contoh:

  .venv-yolo/Scripts/python.exe src/run_experiments_campaign.py \
      --imgsz 960 --device 0 \
      --merge reports/eval_tta.json \
      --out-prefix reports/experiments_v2_master

Outputs:
  reports/experiments_v2_master.json  (matriks variants x datasets + delta + verdict)
  reports/experiments_v2_master.csv   (long form: satu baris per variant x dataset)
  reports/experiments_v2_master.html  (tabel ringkas)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Impor ulang logika evaluasi & konstanta — JANGAN reimplement.
from eval_detection import DATASETS, count_images, evaluate_one, resolve_val_dirs
from common import write_csv, write_json

try:
    # Reuse ambang verdict dari compare_ab bila importable.
    from compare_ab import NEUTRAL_EPS
except Exception:  # pragma: no cover - fallback bila import gagal
    NEUTRAL_EPS = 0.005

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "map50_95"
DATASET_IDS = [d["id"] for d in DATASETS]
DATASET_DISPLAY = {d["id"]: d["display"] for d in DATASETS}
DATASET_INDOMAIN = {d["id"]: d["in_domain"] for d in DATASETS}

# Pemetaan stem file --merge -> nama varian, agar JSON pra-hitung terlipat ke baris yang benar.
MERGE_STEM_TO_VARIANT = {
    "eval_tta": "tta",
    "eval_baseline": "baseline",
    "eval_mowa": "mowa_1pass",
    "eval_mowa_ft": "mowa_1pass_ft",
}


def build_variants(weights_baseline: Path, weights_ft: Path) -> List[Dict]:
    """Tabel varian. `source`='eval' -> jalankan evaluate_one; 'merge' -> ambil dari JSON.

    rectified_root None berarti evaluasi gambar ASLI (baseline / radial_retrain).
    merge_default = path JSON pra-hitung yang otomatis dicek walau tanpa --merge (mis. TTA).
    """
    radial_ft = ROOT / "train model" / "runs_radial" / "ft_radial_yolov8m" / "weights" / "best.pt"
    return [
        {
            "name": "baseline",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": None,
            "merge_default": None,
            "note": "Bobot baseline, gambar asli (referensi Δ).",
        },
        {
            "name": "mowa_1pass",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "rectified",
            "merge_default": None,
            "note": "MOWA 1 pass, bobot baseline.",
        },
        {
            "name": "mowa_1pass_ft",
            "source": "eval",
            "weights": weights_ft,
            "rectified_root": ROOT / "data" / "rectified",
            "merge_default": None,
            "note": "MOWA 1 pass, bobot fine-tune-on-rectified (rectify-both).",
        },
        {
            "name": "mowa_iter2",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "rectified_iter2",
            "merge_default": None,
            "note": "MOWA 2 pass iteratif (Unit 3). Skip bila dir belum ada.",
        },
        {
            "name": "enhanced",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "enhanced",
            "merge_default": None,
            "note": "CLAHE+unsharp preprocessing (Unit 4). Skip bila dir belum ada.",
        },
        {
            "name": "tta",
            "source": "merge",
            "weights": weights_baseline,
            "rectified_root": None,
            "merge_default": ROOT / "reports" / "eval_tta.json",
            "note": "TTA multi-scale+flip. Diproduksi eval_detection_tta.py, di-merge dari JSON.",
        },
        {
            "name": "radial_retrain",
            "source": "eval",
            "weights": radial_ft,
            "rectified_root": None,
            "merge_default": None,
            "note": "Retrain dengan augmentasi distorsi radial (Unit 7), gambar asli.",
        },
        # --- Varian MOWA "soften/coarse/jaga-tepi" (catatan dosen #3/#4). Fase SCREEN
        # memakai bobot baseline (tanpa retrain): ukur apakah pelemahan warp menurunkan
        # penalti vs mowa_1pass (-0.053). Skip bila dir rectified belum digenerasi. ---
        {
            "name": "mowa_coarse",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "rectified_coarse",
            "merge_default": None,
            "note": "MOWA coarse-only (TPS, buang residual flow), bobot baseline.",
        },
        {
            "name": "mowa_alpha05",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "rectified_alpha05",
            "merge_default": None,
            "note": "MOWA warp-alpha=0.5 (perlemah warp), bobot baseline.",
        },
        {
            "name": "mowa_pad015",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "rectified_pad015",
            "merge_default": None,
            "note": "MOWA pad-frac=0.15 (jaga tepi, box tak dibuang), bobot baseline.",
        },
        {
            "name": "mowa_coarse_sam",
            "source": "eval",
            "weights": weights_baseline,
            "rectified_root": ROOT / "data" / "rectified_coarse_sam",
            "merge_default": None,
            "note": "MOWA coarse-only + label bbox via mask SAM (Lever B), bobot baseline.",
        },
        {
            # Pemenang screening di-RETRAIN rectify-both 40 epoch (kondisi B'-pad).
            "name": "mowa_pad015_ft",
            "source": "eval",
            "weights": ROOT / "train model" / "runs_pad015" / "ft_pad015_yolov8m" / "weights" / "best.pt",
            "rectified_root": ROOT / "data" / "rectified_pad015",
            "merge_default": None,
            "note": "MOWA pad-frac=0.15 + fine-tune-on-rectified 40ep (rectify-both, jaga tepi).",
        },
    ]


def load_merge_jsons(paths: List[Path]) -> Dict[str, Dict[str, Dict]]:
    """Muat JSON pra-hitung -> {variant_name: {ds_id: metrics}}.

    Nama varian ditebak dari stem file (MERGE_STEM_TO_VARIANT); stem tak dikenal
    memakai stem itu sendiri sebagai nama varian.
    """
    merged: Dict[str, Dict[str, Dict]] = {}
    for p in paths:
        if not p.exists():
            print(f"[merge] LEWATI (tidak ada): {p}", file=sys.stderr)
            continue
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover
            print(f"[merge] gagal baca {p}: {exc}", file=sys.stderr)
            continue
        variant = MERGE_STEM_TO_VARIANT.get(p.stem, p.stem)
        merged[variant] = extract_dataset_metrics(payload)
        print(f"[merge] {p.name} -> varian '{variant}' ({len(merged[variant])} dataset)")
    return merged


def extract_dataset_metrics(payload: Dict) -> Dict[str, Dict]:
    """Ambil metrik ringkas per dataset dari payload eval_detection (indeks by id)."""
    out: Dict[str, Dict] = {}
    for d in payload.get("datasets", []):
        out[d["id"]] = {
            "status": d.get("status"),
            "images": d.get("images"),
            "map50": d.get("map50"),
            "map50_95": d.get("map50_95"),
            "precision": d.get("precision"),
            "recall": d.get("recall"),
        }
    return out


def run_eval_variant(variant: Dict, imgsz: int, device: str) -> Dict:
    """Jalankan evaluate_one untuk varian 'eval' pada 3 dataset. Guard input hilang."""
    weights: Optional[Path] = variant["weights"]
    rectified_root: Optional[Path] = variant["rectified_root"]

    if weights is None or not Path(weights).exists():
        return {"status": "missing_weights", "datasets": {}}
    if rectified_root is not None and not Path(rectified_root).is_dir():
        return {"status": "missing_input", "datasets": {}}

    ds_metrics: Dict[str, Dict] = {}
    for ds in DATASETS:
        val_dirs = resolve_val_dirs(ds, rectified_root)
        n = count_images(val_dirs)
        print(f"[eval] {variant['name']} :: {ds['id']} <- {[str(d) for d in val_dirs]} (n={n})")
        res = evaluate_one(Path(weights), ds, val_dirs, imgsz, device)
        ds_metrics[ds["id"]] = {
            "status": res.get("status"),
            "images": res.get("images"),
            "map50": res.get("map50"),
            "map50_95": res.get("map50_95"),
            "precision": res.get("precision"),
            "recall": res.get("recall"),
        }
        if res.get("status") == "ok":
            print(f"   mAP50={res['map50']:.4f}  mAP50-95={res['map50_95']:.4f} (n={res['images']})")
        else:
            print(f"   SKIP ({res.get('status')})")
    ok = any(m.get("status") == "ok" for m in ds_metrics.values())
    return {"status": "ok" if ok else "no_data", "datasets": ds_metrics}


def compute_deltas(variant_ds: Dict[str, Dict], baseline_ds: Dict[str, Dict]) -> Dict:
    """Δ mAP50-95 per dataset vs baseline + mean Δ + verdict (dead-band NEUTRAL_EPS)."""
    per_ds: List[Dict] = []
    deltas: List[float] = []
    for ds_id in DATASET_IDS:
        bm = baseline_ds.get(ds_id, {})
        vm = variant_ds.get(ds_id, {})
        b = bm.get(PRIMARY)
        v = vm.get(PRIMARY)
        if isinstance(b, (int, float)) and isinstance(v, (int, float)):
            d = round(v - b, 5)
            deltas.append(d)
            label = "better" if d > NEUTRAL_EPS else "worse" if d < -NEUTRAL_EPS else "neutral"
        else:
            d = None
            label = "n/a"
        per_ds.append({"dataset": ds_id, "delta": d, "label": label})
    mean_delta = round(sum(deltas) / len(deltas), 5) if deltas else None
    if mean_delta is None:
        overall = "unknown"
    elif mean_delta > NEUTRAL_EPS:
        overall = "better"
    elif mean_delta < -NEUTRAL_EPS:
        overall = "worse"
    else:
        overall = "neutral"
    return {
        "per_dataset": per_ds,
        "mean_delta_primary": mean_delta,
        "overall": overall,
        "n_better": sum(1 for p in per_ds if p["label"] == "better"),
        "n_worse": sum(1 for p in per_ds if p["label"] == "worse"),
        "n_neutral": sum(1 for p in per_ds if p["label"] == "neutral"),
    }


# ---------------------------------------------------------------------------- #
# Output writers
# ---------------------------------------------------------------------------- #

def build_long_rows(results: List[Dict]) -> List[Dict]:
    """Long form: satu baris per (variant, dataset)."""
    rows: List[Dict] = []
    for r in results:
        vname = r["name"]
        delta_by_ds = {p["dataset"]: p for p in r.get("verdict", {}).get("per_dataset", [])}
        for ds_id in DATASET_IDS:
            m = r["datasets"].get(ds_id, {})
            dv = delta_by_ds.get(ds_id, {})
            rows.append({
                "variant": vname,
                "variant_status": r["status"],
                "dataset": ds_id,
                "in_domain": DATASET_INDOMAIN.get(ds_id),
                "images": m.get("images"),
                "map50": m.get("map50"),
                "map50_95": m.get("map50_95"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "ds_status": m.get("status"),
                "delta_map50_95": dv.get("delta"),
                "delta_label": dv.get("label"),
            })
    return rows


def _fmt(v: Optional[float]) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) else "—"


def _delta_cell(v: Optional[float]) -> str:
    if not isinstance(v, (int, float)):
        return '<td style="color:#888">—</td>'
    color = "#0a0" if v > NEUTRAL_EPS else "#c00" if v < -NEUTRAL_EPS else "#888"
    sign = "+" if v >= 0 else ""
    return f'<td style="color:{color};font-weight:600">{sign}{v:.4f}</td>'


VERDICT_COLOR = {"better": "#0a0", "worse": "#c00", "neutral": "#888",
                 "unknown": "#888", "baseline": "#333"}


def write_html(path: Path, results: List[Dict], meta: Dict) -> None:
    """Tabel ringkas: baris varian, per-dataset mAP50-95 + Δ, mean Δ, verdict."""
    ds_headers = "".join(
        f'<th>{html.escape(DATASET_DISPLAY[i])}<br>mAP50-95</th><th>Δ</th>' for i in DATASET_IDS
    )
    trs: List[str] = []
    for r in results:
        delta_by_ds = {p["dataset"]: p for p in r.get("verdict", {}).get("per_dataset", [])}
        cells = [
            f'<td style="text-align:left">{html.escape(r["name"])}</td>',
            f'<td>{html.escape(r["status"])}</td>',
        ]
        for ds_id in DATASET_IDS:
            m = r["datasets"].get(ds_id, {})
            cells.append(f"<td>{_fmt(m.get(PRIMARY))}</td>")
            cells.append(_delta_cell(delta_by_ds.get(ds_id, {}).get("delta")))
        vd = r.get("verdict", {})
        overall = vd.get("overall", "unknown")
        color = VERDICT_COLOR.get(overall, "#888")
        cells.append(f"<td>{_fmt(vd.get('mean_delta_primary'))}</td>")
        cells.append(f'<td style="color:{color};font-weight:700">{overall.upper()}</td>')
        trs.append("<tr>" + "".join(cells) + "</tr>")

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Eksperimen V2 — Master MOWA</title>
<style>body{{font-family:Arial,sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:6px 8px;font-size:13px;text-align:center}}th{{background:#f3f3f3}}
h1{{margin-bottom:4px}}code{{background:#f3f3f3;padding:1px 4px}}</style></head><body>
<h1>Tabel Master Eksperimen MOWA V2</h1>
<p>Bobot baseline: <code>{html.escape(str(meta.get("weights_baseline")))}</code> ·
metrik primer = <b>{PRIMARY}</b> · dead-band Δ = {NEUTRAL_EPS} ·
imgsz={meta.get("imgsz")} · device={html.escape(str(meta.get("device")))}</p>
<table><tr><th style="text-align:left">Varian</th><th>Status</th>{ds_headers}<th>mean Δ</th><th>Verdict</th></tr>
{''.join(trs)}
</table>
<p style="color:#666;font-size:12px;margin-top:12px">Δ = varian − baseline pada mAP50-95;
hijau = lebih baik dari baseline, merah = lebih buruk, abu = dalam dead-band {NEUTRAL_EPS}.
Verdict = rata-rata Δ lintas dataset. Status selain 'ok' berarti input varian belum tersedia
(missing_weights / missing_input / external / no_data) sehingga tidak dievaluasi.</p>
</body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")


def print_summary(results: List[Dict]) -> None:
    print("\n=== RINGKASAN KAMPANYE EKSPERIMEN V2 ===")
    header = f"{'variant':16s} {'status':16s} " + " ".join(f"{i[:10]:>10s}" for i in DATASET_IDS) + f" {'mean_d':>8s}  verdict"
    print(header)
    print("-" * len(header))
    for r in results:
        cells = []
        for ds_id in DATASET_IDS:
            v = r["datasets"].get(ds_id, {}).get(PRIMARY)
            cells.append(f"{v:>10.4f}" if isinstance(v, (int, float)) else f"{'—':>10s}")
        vd = r.get("verdict", {})
        md = vd.get("mean_delta_primary")
        md_s = f"{md:>+8.4f}" if isinstance(md, (int, float)) else f"{'—':>8s}"
        print(f"{r['name']:16s} {r['status']:16s} " + " ".join(cells) + f" {md_s}  {vd.get('overall','')}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Orkestrator re-test varian MOWA -> tabel master.")
    ap.add_argument("--weights-baseline", type=Path,
                    default=ROOT / "train model" / "runs_compare" / "cmp_yolov8m" / "weights" / "best.pt")
    ap.add_argument("--weights-ft", type=Path,
                    default=ROOT / "train model" / "runs_rectified" / "ft_rectified_yolov8m" / "weights" / "best.pt")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--device", default="0", help="'0' untuk cuda:0, 'cpu' untuk CPU.")
    ap.add_argument("--variants", default="all",
                    help="Daftar varian dipisah koma (default 'all').")
    ap.add_argument("--merge", nargs="*", type=Path, default=[],
                    help="JSON eval pra-hitung untuk dilipat (mis. reports/eval_tta.json).")
    ap.add_argument("--out-prefix", type=Path, default=ROOT / "reports" / "experiments_v2_master")
    args = ap.parse_args()

    variants = build_variants(args.weights_baseline, args.weights_ft)

    # Filter varian yang diminta. `baseline` selalu diikutkan sebagai acuan Δ,
    # kalau tidak semua verdict akan 'unknown' (tidak ada pembanding).
    if args.variants.strip().lower() != "all":
        wanted = {v.strip() for v in args.variants.split(",") if v.strip()}
        wanted.add("baseline")
        variants = [v for v in variants if v["name"] in wanted]
        if len(variants) <= 1:  # hanya baseline yang tersisa -> tak ada varian valid diminta
            print(f"ERROR: tidak ada varian cocok dengan {sorted(wanted - {'baseline'})}", file=sys.stderr)
            return 2

    # Kumpulkan JSON merge: yang eksplisit dari --merge + merge_default tiap varian.
    merge_paths = list(args.merge)
    for v in variants:
        if v.get("merge_default"):
            merge_paths.append(Path(v["merge_default"]))
    # dedup jaga urutan
    seen = set()
    merge_paths = [p for p in merge_paths if not (str(p) in seen or seen.add(str(p)))]
    merged_data = load_merge_jsons(merge_paths)

    # Proses tiap varian. Baseline diproses dulu agar jadi acuan Δ.
    variants.sort(key=lambda v: 0 if v["name"] == "baseline" else 1)

    results: List[Dict] = []
    baseline_ds: Dict[str, Dict] = {}
    for v in variants:
        name = v["name"]
        # Data merge (bila ada) diutamakan — coordinator bisa pra-hitung tanpa GPU re-run.
        if name in merged_data:
            ds_metrics = merged_data[name]
            status = "ok" if any(m.get("status") == "ok" or isinstance(m.get(PRIMARY), (int, float))
                                 for m in ds_metrics.values()) else "no_data"
            source = "merge"
        elif v["source"] == "merge":
            # Varian merge-only (mis. TTA) tanpa JSON tersedia -> eksternal.
            ds_metrics = {}
            status = "external"
            source = "merge"
        else:
            run = run_eval_variant(v, args.imgsz, args.device)
            ds_metrics = run["datasets"]
            status = run["status"]
            source = "eval"

        rec = {
            "name": name,
            "status": status,
            "source": source,
            "note": v.get("note", ""),
            "weights": str(v["weights"]) if v.get("weights") else None,
            "rectified_root": str(v["rectified_root"]) if v.get("rectified_root") else None,
            "datasets": ds_metrics,
        }
        if name == "baseline":
            baseline_ds = ds_metrics
            rec["verdict"] = {"overall": "baseline", "mean_delta_primary": 0.0,
                              "per_dataset": [{"dataset": i, "delta": 0.0, "label": "baseline"}
                                              for i in DATASET_IDS],
                              "n_better": 0, "n_worse": 0, "n_neutral": 0}
        results.append(rec)

    # Hitung verdict non-baseline setelah baseline_ds terisi.
    for rec in results:
        if rec["name"] == "baseline":
            continue
        rec["verdict"] = compute_deltas(rec["datasets"], baseline_ds)

    # Susun ulang urutan tampilan sesuai definisi build_variants (baseline pertama sudah oke).
    order = {v["name"]: i for i, v in enumerate(build_variants(args.weights_baseline, args.weights_ft))}
    results.sort(key=lambda r: order.get(r["name"], 99))

    meta = {
        "weights_baseline": str(args.weights_baseline),
        "weights_ft": str(args.weights_ft),
        "imgsz": args.imgsz,
        "device": args.device,
        "primary_metric": PRIMARY,
        "neutral_eps": NEUTRAL_EPS,
        "datasets": [{"id": i, "display": DATASET_DISPLAY[i], "in_domain": DATASET_INDOMAIN[i]}
                     for i in DATASET_IDS],
    }
    payload = {"meta": meta, "variants": results}

    out_prefix: Path = args.out_prefix
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_prefix.with_suffix(".json"), payload)
    write_csv(out_prefix.with_suffix(".csv"), build_long_rows(results))
    write_html(out_prefix.with_suffix(".html"), results, meta)

    print_summary(results)
    print(f"\n[campaign] tulis {out_prefix.with_suffix('.json')} / .csv / .html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
