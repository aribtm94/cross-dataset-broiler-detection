# Third-Party Licenses & Attributions

This project bundles or depends on external code, model weights, datasets, and
reference material. The original code in this repository is MIT-licensed (see
[`LICENSE`](LICENSE)), but the components below carry **their own terms**. The
most important constraint: **MOWA is non-commercial**, so the end-to-end
pipeline using the bundled MOWA weights is for **research/education only**.

| Component | Where | License | Use restriction |
|-----------|-------|---------|-----------------|
| [MOWA](https://github.com/KangLiao929/MOWA) | `vendor/MOWA` (mowa-sam.zip) | S-Lab License 1.0 | **Non-commercial only** |
| [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) | pip dep + trained `*.pt` | AGPL-3.0 | Copyleft; network use triggers source disclosure |
| [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) | `mobile_sam.pt` (mowa-sam.zip) | Apache-2.0 | Permissive |
| PyTorch / torchvision | pip deps | BSD-3-Clause | Permissive |
| Cobb500 Performance Supplement 2022 | `configs/` + growth constants | © Cobb-Vantress, Inc. | Reference standard, redistribution restricted |
| Roboflow Universe datasets | `data/external/*` (datasets-core.zip) | Per-dataset (see each README) | Varies; check before reuse |
| Reference papers (PDFs) | repo root + `papers/` (papers.zip) | © respective publishers | Convenience copies only |

## MOWA — S-Lab License 1.0 (non-commercial)

MOWA (Multiple-in-One Image Warping, TPAMI 2025) provides the end-to-end
fisheye/wide-angle rectification used as an A/B preprocessing step.

- Source: <https://github.com/KangLiao929/MOWA>
- The full S-Lab License 1.0 text ships inside the vendored repo at
  `vendor/MOWA/LICENSE`.
- **You may not use MOWA or its outputs for commercial purposes.** This is why
  the whole pipeline, as bundled, is scoped to academic use.

## Ultralytics YOLO — AGPL-3.0

Detection uses the Ultralytics framework. The trained checkpoints in
`weights-yolo.zip` are derivative works of AGPL-3.0 software; if you
redistribute them or expose them over a network service, AGPL-3.0 obligations
(including source availability) apply. Ultralytics also offers a commercial
license for closed-source use.

## MobileSAM — Apache-2.0

`mobile_sam.pt` powers mask-guided rectification variants (SAM masks). Apache-2.0
is permissive; retain the license and attribution notices.

## Cobb500 growth standard

The `configs/cobb500_as_hatched.csv` values and the `COBB500_AS_HATCHED` table in
`src/common.py` are transcribed from the *Cobb500 Broiler Performance & Nutrition
Supplement (2022)*, © Cobb-Vantress, Inc. Used only as a reference growth curve
for relative weight estimation.

## External datasets

External datasets are pulled from Roboflow Universe and are described in
`configs/datasets/external_datasets.json`. Each downloaded dataset keeps its own
`README.roboflow.txt` / license under `data/external/<dataset>/`. Check those
terms before redistributing dataset images.

## Reference papers

The PDF files (`FisheyeDetNet…`, `FisheyeYOLO…`, `KITTI360…`, `WoodScape…` at the
repo root, and everything in `papers/`) are third-party publications included
only to document the methodology. Copyright remains with the original
publishers; they are not covered by this repository's MIT license.

---

If you believe any attribution here is incomplete or incorrect, please open an
issue so it can be fixed.
