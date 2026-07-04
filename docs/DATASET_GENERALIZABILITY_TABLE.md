# Dataset Comparison Table — Generalizability Study

Tanggal: 2026-06-27

## Ringkasan Dataset

| Dataset | Source | Public/downloaded | Format final | Images | Valid bbox | Median bbox/img | Age/week available | Recommended use |
|---|---|---:|---|---:|---:|---:|---:|---|
| PIO baseline | Zenodo / Scientific Data 2026 | ✅ | YOLO bbox | 1,487 | 327,283 | 179.0 | ✅ | Baseline absolute Cobb500 mode |
| NESTLER | Zenodo | ✅ | Video JSON → YOLO bbox | 480 sampled | 4,043 | 9.0 | ❌ | Relative anomaly, sparse/medium density domain shift |
| Broiler Healthy & Sick | Roboflow | ✅ | YOLO-seg → YOLO bbox | 491 | 491 | 1.0 | ❌ | Limited robustness check; not valid for flock density |
| Broiler Instance Segmentation | Roboflow | ✅ | YOLO-seg → YOLO bbox | 200 | 10,570 | 53.0 | ❌ | Strong external broiler dense test |
| Chicken Count | Roboflow | ✅ | YOLO bbox | 178 | 3,646 | 18.0 | ❌ | Counting/domain-shift test, mixed resolution |
| FUM Chicken Detection | Roboflow | ✅ | YOLO bbox | 326 | 29,355 | 88.5 | ❌ | Strong external dense chicken detection test |

## Suitability Scoring

| Dataset | Detection suitability | Weight-estimation suitability | Relative-anomaly suitability | Generalizability value | Notes |
|---|---|---|---|---|---|
| PIO baseline | High | High (relative Cobb500) | High | Reference dataset | Has week metadata; use for absolute mode |
| NESTLER | Medium | Low | Medium | Medium | Sparse/medium density, video frames, no age |
| Broiler Healthy & Sick | Medium | Low | Low | Low | Mostly one chicken/image; useful only to show pipeline robustness |
| Broiler Instance Segmentation | High | Low | High | High | Broiler-specific dense dataset; strong evidence source |
| Chicken Count | Medium-High | Low | Medium | Medium-High | Mixed resolution causes stronger domain shift |
| FUM Chicken Detection | High | Low | High | High | Dense dataset; good test of high-density generalization |

## Relative Anomaly Results

| Dataset | P97+ candidate | P99+ critical | Image CV median | Interpretation |
|---|---:|---:|---:|---|
| NESTLER | 3.02% | 1.01% | 50.15 | Percentile threshold stable; high image variation |
| Broiler Healthy & Sick | 0.00% | 0.00% | 0.00 | Sparse dataset; per-image anomaly not meaningful |
| Broiler Instance Segmentation | 3.78% | 1.89% | 22.96 | Good relative-anomaly generalization case |
| Chicken Count | 3.73% | 2.58% | 22.82 | Domain shift/mixed resolution increases critical rate |
| FUM Chicken Detection | 3.52% | 1.48% | 34.82 | Good dense detection generalization case |

## Recommended Dataset Tiers

### Tier A — Strong for thesis generalizability

1. `broiler_instance_seg`
2. `chicken_detection_fum`
3. `PIO baseline`

Reason: dense enough, many bbox per image, relevant for detection/generalization.

### Tier B — Useful domain-shift / robustness checks

1. `nestler_yolo`
2. `chicken_count`

Reason: different domain, lower/mixed density, useful to show limits.

### Tier C — Limited-use only

1. `broiler_healthy_sick`

Reason: mostly single object per image, unsuitable for flock/image-context anomaly.

## Key Sentence for Report

> Public external poultry datasets rarely provide age or actual body-weight ground truth, therefore the external evaluation is framed as relative visual-anomaly generalization rather than absolute weight prediction.
