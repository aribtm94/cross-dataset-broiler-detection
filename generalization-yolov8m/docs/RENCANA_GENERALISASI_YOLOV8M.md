# Rencana Metodologi — Meningkatkan Robustness & Generalisasi YOLOv8m (PIO → dataset lain)

Tanggal: 2026-07-02 · **Revisi arah: 2026-07-06** (retrain PIO-only + augmentasi oklusi jadi jalur utama — lihat §11–§14).
Status: **rencana + pustaka referensi** (belum diimplementasikan; sesuai instruksi "buat rencana dulu").
Sumber: PIO paper (`s41597-026-07114-5`) + hasil deep-research (27 sumber, 25 klaim diverifikasi 3-vote, 23 confirmed / 2 refuted).

Keputusan scoping yang sudah dikunci bersama user:
1. **Rezim latih = zero-shot / single-source (PIO saja).** Tidak boleh pakai gambar 3 dataset eksternal saat training. Leverage = training recipe (augmentasi) + trik inference, bukan target data. **Augmentasi oklusi mematuhi ini** (disintesis dari PIO, bukan mengintip target).
2. **Compute — REVISI 2026-07-06:** training pindah ke **PC lain ber-RTX 5090 (32GB, Blackwell)** → **retrain penuh feasible** (imgsz 960, batch besar, banyak run). Metode **inference-time no-retrain** (SAHI/TTA/kalibrasi) **tidak dibuang** — turun status jadi *pelengkap* yang bisa ditumpuk di atas model hasil retrain. (Constraint lama CPU + RTX 3050 6GB / RAM ~3.5GB hanya berlaku untuk mesin lokal saat **eval**, bukan lagi untuk training.)
3. **Target = keduanya** — dekati/lewati angka in-domain PIO **dan** bangun cerita generalisasi lintas-dataset (paper PIO tidak punya ini sama sekali).
4. **Jalur utama baru:** **retrain PIO-only dengan augmentasi oklusi + resep DG** (§11–§14), dievaluasi dengan protokol fair §7 yang tidak berubah.

---

## 0. TL;DR (baca ini dulu)

- **Yang benar-benar "mengalahkan paper" bukan angka mAP in-domain** (kamu sudah imbang/menang di recall & mAP50-95; gap di mAP50 0.936 vs 0.97 kemungkinan **bukan apple-to-apple**). **Yang mengalahkan paper adalah studi generalisasi lintas-dataset** — sesuatu yang paper PIO **sama sekali tidak lakukan**.
- Di bawah zero-shot + CPU, hanya **3 keluarga metode yang benar-benar no-retrain**: **(1) SAHI** (paling kuat, untuk FUM & broiler dense), **(2) TTA** (`augment=True`), **(3) kalibrasi threshold conf/NMS** di-tune pada PIO-val yang diberi perturbasi (bebas target data).
- **FUM (mAP50 0.139) bukan model buta** — itu **annotation-protocol / localization gap**. Literatur mengkonfirmasi ini sebagai penyebab kegagalan cross-domain yang *terpisah* dari domain shift visual. Solusinya sebagian **memperbaiki deteksi (SAHI + threshold)**, sebagian **memperbaiki cara mengukur (multi-IoU + dekomposisi TIDE + P/R di operating point)**.
- **NESTLER (mAP50 0.0)** = shift viewpoint top-down→side-view = kasus **terberat**. Zero-shot ceiling-nya rendah secara fundamental. **Framing jujur sebagai batas generalisasi**, bukan bug yang harus di-nol-kan.
- **Retrain (dulu "Tier 2") kini jalur UTAMA** karena training pindah ke **RTX 5090** (§11–§14). Recipe DG (augmentasi diversifikasi + **oklusi** + normalization-perturbation) memberi gain **single-digit mAP** di literatur (bukan target data). ⚠️ **Oklusi menyerang hanya satu mekanisme** — visibilitas parsial di scene padat: bantu sumbu PIO↔broiler↔FUM, **~nol** untuk viewpoint nestler & anotasi FUM (§12). Jangan bingkai oklusi sebagai peratas semua dataset.

---

## 1. Situasi saat ini (baseline jujur)

### 1.1 In-domain PIO val — paper vs punyamu

| Metric | Paper YOLOv10m | **YOLOv8m (punyamu)** | Verdict |
|---|---|---|---|
| Precision | 0.961 | 0.958 | ~imbang |
| Recall | 0.88 | **0.888** | **menang** |
| mAP@50 | **0.97** | 0.936 | paper lebih tinggi |
| mAP@50-95 | 0.76 | **0.769** | **menang** |

Catatan kritis soal angka 0.97 paper: lompat dari 0.92 (v10s) → 0.97 (v10m) padahal mAP50-95 hanya 0.70→0.76; dibulatkan 2 desimal; kemungkinan split 70/30 berbeda. **Jangan kejar 0.97 seolah itu angka keramat** — kemungkinan besar tidak sebanding langsung. (Sumber: PIO paper Table 8.)

### 1.2 Cross-dataset zero-shot (baseline saat ini — dieval pakai best.pt yolo11m; harus diulang dgn yolo8m)

| Dataset | Domain | Resolusi | Median bbox/img | mAP50 | Diagnosis |
|---|---|---|---|---|---|
| broiler_instance_seg | top-down dense (mirip PIO) | 640×640 | 53 | **0.836** | bagus, in-domain-like |
| chicken_detection_fum | dense sangat padat | **1920×1080** | 88.5 | **0.139** | *mendeteksi* tapi box tak capai IoU≥0.5 + over-detect → **localization/annotation gap** |
| nestler_yolo | side-view backyard | 1920×1080 | 9 | **0.0** | **total failure**: viewpoint shift berat |

> ⚠️ **Tindakan wajib pertama:** baseline di atas dihitung dengan `best.pt` **yolo11m**. Karena fokus tesis = **yolo8m**, ulang eval memakai `PIO/runs_compare/cmp_yolov8m/weights/best.pt`. Angka yolo8m ≈ yolo11m (metrik PIO nyaris identik) tapi harus dilaporkan dari model yang benar.

### 1.3 Insight resolusi (penting untuk memilih metode)

- **FUM 1920×1080 + 88 box/img**: pada imgsz 960 gambar diciutkan ~2× → ayam kecil jadi makin kecil. **Ini kasus ideal SAHI** (slice jadi ubin 960, kembalikan resolusi asli).
- **broiler 640×640**: sudah kecil (di-resize Roboflow) → SAHI **tidak banyak menolong**; lagipula sudah 0.836.
- **nestler 1920×1080**: masalahnya **viewpoint**, bukan resolusi → SAHI tak menyelamatkan.

---

## 2. Kerangka ilmiah: ini masalah **Single-Domain Generalized Object Detection (S-DGOD)**

Setting-mu persis definisi formal S-DGOD: **latih di SATU source domain, tanpa target data, generalisasi ke banyak unseen target.** Ini melegitimasi framing tesis.

- **Wu & Deng, CVPR 2022** — *Single-Domain Generalized Object Detection in Urban Scene via Cyclic-Disentangled Self-Distillation* (CDSD). Memperkenalkan problem & benchmark S-DGOD. [PDF](https://openaccess.thecvf.com/content/CVPR2022/papers/Wu_Single-Domain_Generalized_Object_Detection_in_Urban_Scene_via_Cyclic-Disentangled_Self-Distillation_CVPR_2022_paper.pdf) · code: github.com/AmingWu/Single-DGOD
- **Survey 2026** — *Generalization Under Scrutiny: Cross-Domain Detection Progresses, Pitfalls, and Persistent Challenges* (arXiv:2604.08230). DG untuk deteksi "under-investigated relatif ke UDA tapi lebih applicable" saat target data langka/tak ada. [PDF](https://arxiv.org/pdf/2604.08230)

**Kalimat framing untuk proposal/sidang:**
> Penelitian ini mengevaluasi detektor broiler yang dilatih hanya pada PIO dalam setting *single-domain generalization*: model diuji zero-shot pada dataset ayam lain tanpa pernah melihat gambar target saat training, lalu ditingkatkan robustness-nya dengan metode inference-time dan recipe augmentasi bebas-target.

---

## 3. Pustaka referensi (citation library) — siap dikutip

### Sub-topik 1 — DG / S-DGOD (feature-norm & style)
- **Normalization Perturbation (NP)**, ICLR 2023 — arXiv:2211.04393. Perturbasi statistik channel fitur shallow untuk sintesis "latent styles"; **single source, tanpa target data**. Train-time, custom code. [PDF](https://arxiv.org/pdf/2211.04393)
- **SDG-YOLOv8**, 2024 — ScienceDirect S0141938224003123. **Berbasis YOLOv8!** Dua modul: (a) local-global transformation (bikin auxiliary domain, anotasi tetap), (b) normalization-perturbation fusion di feature space. Perlu retrain + modul + loss baru. Paling relevan sebagai *recipe* untuk tesis. [link](https://www.sciencedirect.com/science/article/abs/pii/S0141938224003123)
- **DivAlign**, CVPR 2024 — arXiv:2405.14497. Single-source (diversify + align). **Kuantifikasi ekspektasi gain DG: +3.6 s/d +8.4 mAP@0.5** (angka dari urban weather, bukan ayam). [abs](https://arxiv.org/abs/2405.14497)

### Sub-topik 2 — Augmentasi untuk robustness
Recipe augmentasi (HSV/photometric, blur, noise, CLAHE, mosaic, mixup, copy-paste, multi-scale) adalah leverage utama zero-shot bawaan Ultralytics. Bukti kuantitatif spesifik "berapa gain tiap augmentasi" **tidak** lolos verifikasi sebagai angka pasti → laporkan sebagai *ablation empiris milikmu*, bukan klaim dari paper. Payung teori: NP & SDG-YOLOv8 (feature-level) + DivAlign (image-level diversification).

### Sub-topik 3 — SAHI (prioritas #1 no-retrain)
- **Akyon et al., ICIP 2022** — *Slicing Aided Hyper Inference* (arXiv:2202.06934). "Dapat diterapkan di atas detektor apapun tanpa fine-tuning." Gain inference-only terukur: **+6.8% (FCOS), +5.1% (VFNet), +5.3% (TOOD) AP** di data small-object (VisDrone/xView). [abs](https://arxiv.org/abs/2202.06934)
- **Ultralytics SAHI guide** — jalan di YOLO pretrained, "no model modifications or retraining", **hemat VRAM** (cocok 6GB). [docs](https://docs.ultralytics.com/guides/sahi-tiled-inference)
- ⚠️ SAHI **menambah latency** (N slice ≈ N forward pass + merge-NMS) dan **bisa memperparah over-detect FUM** kalau param merge (overlap ratio, postprocess IoU) tak di-tune.

### Sub-topik 4 — TTA
- **Ultralytics YOLOv5 TTA** — flip + multi-scale, merge sebelum NMS; `augment=True`; **biaya ~2-3× inference**. [docs](https://docs.ultralytics.com/yolov5/tutorials/test-time-augmentation)
- ⚠️ **Besar gain-nya TIDAK pasti** (lihat §8 — klaim "+1.2 mAP COCO" **direfutasi**). Perlakukan TTA sebagai opsi murah dengan manfaat *tak-terukur untuk ayam* → ukur sendiri.

### Sub-topik 5 — Kalibrasi threshold conf/NMS lintas domain
- **Tomani et al., CVPR 2021 (Oral)** — *Post-Hoc Uncertainty Calibration for Domain Drift* (arXiv:2012.10988). **Kalibrasi post-hoc di validation set yang diberi perturbasi Gaussian 10 level** (bebas target data), hanya men-tune kalibrator, bukan bobot. [PDF](https://openaccess.thecvf.com/content/CVPR2021/papers/Tomani_Post-Hoc_Uncertainty_Calibration_for_Domain_Drift_Scenarios_CVPR_2021_paper.pdf) · *caveat: paper klasifikasi, dipakai secara analogi untuk set threshold conf detektor.*
- **TCD (Munir et al., NeurIPS 2022)** — arXiv:2209.07601. Auxiliary loss samakan confidence↔IoU. **Train-time (perlu retrain)** → masuk Tier 2, bukan no-retrain.

### Sub-topik 6 — Annotation-protocol / label-granularity mismatch (kunci menjelaskan FUM)
- **Survey 2604.08230** — "Annotation bias… different box tightness, different class granularity… central to reliability", dan **"mAP non-decomposable"** (mencampur klasifikasi vs lokalisasi). Rekomendasi: metrik stage-wise (proposal recall, cls-given-oracle-box, localization error, ECE). [PDF](https://arxiv.org/pdf/2604.08230)
- **Bridging Annotation Gaps**, 2025 — arXiv:2506.04737. Transfer/gabung dataset naif → "annotation conflicts in class semantics, labelling granularity, bbox styles". Metode LAT **+4.2/+4.8 AP** dengan memperbaiki mismatch anotasi. [html](https://arxiv.org/html/2506.04737v2)
- **Can We Trust Bounding Box Annotations?**, CVPRW 2022 — ganti *konvensi* anotasi (seg-induced box vs human box) **menurunkan AP75 ~6.3% & AP50:95 ~7.6-9.3% TANPA mengubah detektor**. Bukti konkret bahwa protokol label saja bisa menekan mAP. [PDF](https://openaccess.thecvf.com/content/CVPR2022W/VDU/papers/Murrugarra-Llerena_Can_We_Trust_Bounding_Box_Annotations_for_Object_Detection_CVPRW_2022_paper.pdf)
- **TIDE (Bolya et al., ECCV 2020)** — dekomposisi (1−mAP) jadi error klasifikasi vs lokalisasi. Alat wajib untuk membuktikan "FUM bukan buta". 
- **Sensitivity of AP to BBox Perturbations** — arXiv:2206.10107. Geser box 5-10px ubah mAP ~10%.

### Sub-topik 7 — Viewpoint / domain shift berat (untuk NESTLER)
- CDSD (CVPR 2022): worst-domain mAP50 **16.6** vs in-domain 56.1 → ceiling zero-shot rendah & sangat variatif; dikonfirmasi silang di PhysAug (arXiv:2412.11807) & G-NAS (arXiv:2402.04672). *Tapi shift CDSD masih same-viewpoint (cuaca), tak sampai 0.0.*
- **Cross-dataset under Domain Specificity**, 2026 — arXiv:2601.09497. Transfer **paling parah saat source setting-specific → target setting-agnostic** — persis PIO (kandang top-down spesifik) → nestler (backyard agnostik).

### Sub-topik 8 — Studi poultry/livestock cross-domain
- **TIDAK ADA** studi generalisasi lintas-dataset khusus ayam/broiler/livestock yang lolos verifikasi. **Ini gap literatur nyata → memperkuat novelty tesismu**, tapi artinya kamu **tak punya baseline same-species langsung** untuk dibandingkan. Framing: "sepengetahuan kami, belum ada studi generalisasi lintas-dataset deteksi broiler; kami yang pertama."

### Sub-topik 9 — Framing tesis defensible
- arXiv:2601.09497 (skema pelaporan dua-protokol), arXiv:2604.08230 (metrik stage-wise), **label convergence** arXiv:2409.09412 (anotasi tak konsisten memberi *upper bound* performa — skor rendah bisa cermin anotasi, bukan model).

---

## 4. Rencana bertingkat

### TIER 0 — Perbaiki baseline & protokol evaluasi (WAJIB, tanpa retrain, murah)
Tujuan: angka yang jujur & fair sebelum menaikkan apapun.
1. **Re-eval dengan yolo8m** (`cmp_yolov8m/weights/best.pt`) di PIO-val + 3 dataset eksternal, batch=1 imgsz=960 (sesuai constraint RAM).
2. **Adopsi pelaporan fair** (bukan mAP50 tunggal):
   - mAP di **multi-IoU**: 0.3 / 0.5 / 0.75 (0.3 mengungkap "terdeteksi tapi box longgar" di FUM).
   - **P/R + F1 di operating point** conf tetap, per dataset.
   - **Dekomposisi TIDE** (cls vs loc error) khusus FUM → bukti kuantitatif "bukan buta".
   - Laporkan **jumlah pred vs GT** (over/under-detection) per dataset.
3. **Overlay GT vs pred** tetap dipertahankan sebagai bukti visual (sudah jadi praktikmu — lihat memory).

### TIER 1 — Inference-time, TANPA retrain (leverage utama)
4. **Kalibrasi threshold conf & NMS-IoU per dataset**, di-tune pada **PIO-val + perturbasi** (Gaussian noise / blur / brightness sweep) — **bebas target label**, à la Tomani 2021. Menekan over-detection FUM yang meng-inflate FP.
5. **SAHI** untuk **FUM** (dan uji di broiler): slice ~960×960, tune overlap ratio + postprocess (NMS/greedy) & match_threshold. Ekspektasi: gain terbesar di FUM 1920×1080. *Ukur; angka +5-7 AP adalah dari domain lain.*
6. **TTA** (`augment=True`) sebagai add-on murah; ukur apakah membantu (jangan asumsikan).
7. **(Opsional) Preprocessing normalization**: CLAHE / resize konsisten untuk menyamakan pencahayaan & skala — murah, uji sebagai ablation.

### TIER 2 — Retrain zero-shot DG (JALUR UTAMA sejak revisi 2026-07-06; tetap PIO-only)
> **Update:** training di **RTX 5090** → bukan lagi "light/opsional". Detail recipe oklusi + ablation + setup ada di **§11–§14**. Prasyarat: torch **cu128** (Blackwell sm_120) — lihat §13.3.
8. **Recipe augmentasi DG kuat** saat retrain di PIO: HSV/photometric agresif, blur, noise, mosaic, mixup, copy-paste, **multi-scale**. Ini implementasi "image-level diversification" (DivAlign/SDG-YOLOv8) via knob bawaan Ultralytics — **tanpa custom code**.
9. **(Advanced, opsional)** Normalization-Perturbation ala NP / SDG-YOLOv8 di backbone — butuh custom code; jadikan eksperimen "stretch" bila waktu ada. Ekspektasi gain single-digit mAP.
10. **(Advanced, opsional)** TCD auxiliary loss saat retrain untuk kalibrasi confidence↔IoU.

### TIER 3 — Framing & limitations (nulis, bukan ngoding)
11. NESTLER dibingkai sebagai **batas viewpoint zero-shot** (didukung §3 sub-topik 7), bukan kegagalan.
12. Klaim novelty: **studi generalisasi lintas-dataset broiler pertama** (gap sub-topik 8).

---

## 5. Strategi per-dataset

| Dataset | Diagnosa | Metode utama | Ekspektasi realistis |
|---|---|---|---|
| **broiler_instance_seg** | in-domain-like, sudah 0.836 | Tier 0 fair-report; SAHI diuji tapi kecil efeknya (640px) | naik tipis; jadi "bukti generalisasi sukses" |
| **chicken_detection_fum** | localization/annotation gap, over-detect | **SAHI + kalibrasi conf/NMS (Tier 1)** + **multi-IoU & TIDE (Tier 0)** | mAP50 naik signifikan **dan** dibuktikan "model tidak buta"; mAP50 tetap dibatasi mismatch anotasi |
| **nestler_yolo** | viewpoint shift berat (top-down→side) | Tier 3 framing; coba Tier 2 augmentasi geometris tapi **jangan berharap** lepas dari ~0 | tetap rendah; jadi "honest ceiling / limitation" |

---

## 6. Matriks eksperimen: dampak × biaya × fit constraint

| # | Teknik | Retrain? | Biaya compute | Fit zero-shot+CPU | Prioritas |
|---|---|---|---|---|---|
| E0 | Re-eval yolo8m + multi-IoU + TIDE | tidak | rendah (CPU ok) | ✅ sempurna | **WAJIB** |
| E1 | Kalibrasi conf/NMS (perturbed PIO val) | tidak | rendah | ✅ | **tinggi** |
| E2 | SAHI (FUM prioritas) | tidak | sedang (latency naik) | ✅ (hemat VRAM) | **tinggi** |
| E3 | TTA `augment=True` | tidak | sedang (2-3×) | ✅ | sedang |
| E4 | Preprocessing (CLAHE/resize) | tidak | rendah | ✅ | sedang |
| E5 | Retrain augmentasi DG (PIO-only) | **ya** | tinggi (butuh CUDA) | ⚠️ perlu GPU setup | sedang (jika GPU) |
| E6 | NP / SDG-YOLOv8 feature-perturb | **ya + custom code** | tinggi | ⚠️ | rendah (stretch) |
| E7 | TCD calibration loss | **ya** | tinggi | ⚠️ | rendah (stretch) |

Aturan main: **selesaikan E0→E4 dulu** (semua no-retrain, muat di CPU). Baru pertimbangkan E5+ jika CUDA di-setup & waktu ada.

---

## 7. Standar pelaporan yang defensible (tabel hasil tesis)

Untuk tiap dataset laporkan: `n_img`, `n_pred`, `n_GT`, `P`, `R`, `F1`, `mAP@0.3`, `mAP@0.5`, `mAP@0.75`, `mAP@0.5:0.95`, dan (khusus FUM) **TIDE cls-error vs loc-error**. Plus overlay GT/pred. Ini mematahkan serangan penguji "kok mAP-nya jelek" karena kamu tunjukkan **di mana** kegagalannya (lokalisasi/anotasi vs deteksi).

Prinsip pemandu (konsisten dgn `REVIEW_METODOLOGI_GENERALISASI.md`): **ukur sesuatu yang BISA gagal.** mAP multi-IoU, over/under-detection ratio, dan TIDE decomposition semuanya bisa keluar jelek → bermakna.

---

## 8. ⚠️ Klaim yang TIDAK boleh dikutip (di-refutasi saat verifikasi)

1. ❌ **"TTA naikkan +1.2 mAP di COCO (0.504→0.516)"** — vote 1-2, gagal verifikasi. → Jangan sebut angka gain TTA; ukur sendiri.
2. ❌ **"Cityscapes 55.2 → nuImages 39.2 / Waymo 44.6 akibat mismatch anotasi"** — vote 0-3, gagal. → Kutip **mekanismenya** (dari 2506.04737 & survey), atau pakai angka **SHBB/AHBB AP75 −6.3%** (CVPRW 2022) yang **lolos** verifikasi.
3. ⚠️ Angka gain **SAHI (+5-7 AP)** dan **DivAlign (+3.6-8.4)** valid tapi **dari domain aerial/urban** — sebut sebagai "ekspektasi orde besaran dari domain lain, divalidasi empiris di data ayam".
4. ⚠️ Tomani 2021 = paper **klasifikasi** dipakai analogi → bilang eksplisit ini analogi, bukan hasil deteksi tervalidasi.

---

## 9. Pertanyaan terbuka (perlu jawaban empiris / keputusan)

1. Berapa gain SAHI & TTA **nyata di FUM/broiler**? Hanya run empiris (no-retrain) yang menjawab — lakukan sebelum mengunci metodologi.
2. Apakah viewpoint NESTLER bisa digeser dari 0.0 **sama sekali** tanpa target data? (uji augmentasi geometris di Tier 2; ekspektasi: hampir tidak).
3. IoU 0.3 atau P/R@operating-point — mana standar pelaporan "fair" yang disepakati pembimbing untuk FUM?
4. Apakah pembimbing setuju **narasi utama = generalisasi** (bukan kejar 0.97 in-domain)?

---

## 10. Urutan eksekusi yang disarankan (checklist — setelah rencana disetujui)

- [ ] **E0**: script re-eval yolo8m + tabel multi-IoU + TIDE (adaptasi `run_external_eval.py`).
- [ ] **E1**: sweep conf/NMS-IoU di PIO-val berperturbasi → pilih operating point → terapkan ke 3 dataset.
- [ ] **E2**: integrasi SAHI (`sahi` pip) untuk FUM; tune slice/overlap/postprocess; bandingkan vs no-SAHI.
- [ ] **E3/E4**: TTA + preprocessing sebagai ablation.
- [ ] Tulis tabel hasil + overlay + narasi §7.
- [ ] (Jika CUDA di-setup) **E5**: retrain PIO-only dgn recipe augmentasi DG; bandingkan generalisasi vs baseline.
- [ ] (Stretch) E6/E7.

> Semua langkah menghormati constraint memory: `PIO/.venv_run`, batch=1 @ imgsz 960, temp di D:. Lihat [[pio-venv-and-run-constraints]] & [[external-dataset-generalization-result]].

---

## 11. REVISI ARAH (2026-07-06) — Jalur retrain PIO-only + augmentasi oklusi

Pemicu: training pindah ke **PC lain ber-RTX 5090 (32GB, Blackwell)** → constraint compute lama (CPU/RTX 3050 6GB) tidak lagi mengikat training. Bentuk oklusi yang dipakai = **augmentasi saat retrain** (bukan occlusion-aware *loss*, bukan sekadar metrik). Tetap PIO-only → framing S-DGOD (§2) utuh; augmentasi = "image-level diversification" (payung DivAlign/SDG-YOLOv8, §3 sub-topik 1). Metode no-retrain (§4 Tier 1: SAHI/TTA/kalibrasi) tetap valid sebagai **pelengkap** yang ditumpuk saat inference di atas model hasil retrain.

## 12. Peta jujur: apa yang oklusi perbaiki vs tidak

| Dataset | Penyebab gap (diagnosa terkunci §1.2) | Efek augmentasi oklusi | Ekspektasi realistis |
|---|---|---|---|
| **broiler** (mAP50 0.836) | in-domain-like, padat top-down | ↑ recall tipis (partial-visibility) | naik kecil → "bukti generalisasi sukses" |
| **FUM** (0.139) | **annotation-protocol / box longgar + over-detect** | ↑ recall mungkin, **TIDAK** memperbaiki lokalisasi/anotasi yang mem-cap mAP50 | mAP50 tetap dibatasi anotasi → **tetap butuh SAHI + kalibrasi + TIDE** (Tier 0/1), bukan oklusi |
| **nestler** (0.0) | **viewpoint shift** (top-down→side) | **~nol** — oklusi tak mengubah pose/sudut | tetap rendah → framing "batas viewpoint" (§4 Tier 3) |

**Kalimat sidang:** *"Augmentasi oklusi diterapkan untuk meningkatkan robustness deteksi pada kondisi visibilitas parsial yang inheren di kandang padat. Kami tidak mengklaim oklusi menutup gap akibat perbedaan protokol anotasi (FUM) atau pergeseran sudut pandang (nestler); keduanya ditangani/dibingkai terpisah."*

**Justifikasi domain-appropriate:** PIO ~220 ayam/gambar, FUM ~88/gambar → oklusi antar-ayam inheren, jadi "melatih deteksi ayam sebagian tertutup" bukan trik asal. Kandidat sitasi (**wajib verifikasi 3-vote dulu**, §14.4): Random Erasing (Zhong dkk. AAAI 2020), Cutout (2017), GridMask (2020), Hide-and-Seek (ICCV 2017), Copy-Paste (Ghiasi dkk. CVPR 2021); related-work crowded detection: Repulsion Loss (CVPR 2018), CrowdDet (CVPR 2020). **Gap literatur = novelty:** belum ada studi oklusi-augmentasi khusus deteksi ayam padat lintas-dataset (konsisten §3 sub-topik 8 / rangkuman §H).

## 13. Recipe oklusi + setup RTX 5090 + jebakan

### 13.1 Knob Ultralytics untuk DETECTION (bukan classification)
> ⚠️ Parameter **`erasing`** = **khusus task classification**; **tidak berlaku** detection. Jalur oklusi detection = `copy_paste`/`mosaic`/`mixup` + albumentations.

| Knob | Peran oklusi | Rentang awal (usulan) | Catatan / risiko |
|---|---|---|---|
| `copy_paste` | tempel ayam → oklusi realistis + padat | 0.1–0.3 | **verifikasi:** efektif penuh butuh **mask segmentasi**; PIO = bbox murni → cek apakah jalan / perlu `copy_paste_mode` atau mask. Jika tak jalan tanpa mask, drop. |
| `mosaic` | oklusi batas antar-tile | 1.0 (default), `close_mosaic=10` | matikan ~10 epoch terakhir agar konvergen ke distribusi asli |
| `mixup` | blend → oklusi transparan | 0.0–0.15 | konservatif di scene padat |
| Albumentations **CoarseDropout** | lubang oklusi kecil | `max_holes` kecil, ukuran lubang **< ukuran 1 ayam** | **kunci anti-ghost-label** (§13.4) |
| Albumentations **GridMask** | oklusi terstruktur | rasio konservatif | bukan default albumentations → perlu tambah transform |

### 13.2 Cara pasang CoarseDropout/GridMask
Ultralytics auto-menerapkan pipeline albumentations default (Blur/MedianBlur/CLAHE/ToGray) **jika `albumentations` terinstall** untuk detection. **Menambah CoarseDropout/GridMask perlu kustomisasi** daftar transform (custom trainer / edit augmentasi) — **verifikasi perilaku versi Ultralytics terbaru**. CoarseDropout mengubah gambar tapi **tidak menghapus label** → maka **ukuran lubang harus kecil** relatif ukuran ayam (oklusi parsial, bukan menghapus ayam utuh).

### 13.3 Setup RTX 5090 (Blackwell, sm_120)
- **Wajib PyTorch build CUDA 12.8 (cu128), torch ≥ 2.7.** Torch lama → error `no kernel image is available for execution on the device` meski `cuda.is_available()`==True.
- `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128` (verifikasi versi terbaru saat eksekusi).
- Cek: `python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`.
- Training di **PC lain** → samakan versi Ultralytics, hyperparameter, **split PIO train/val, seed**; simpan `args.yaml` tiap run agar reproducible & sebanding baseline.

### 13.4 ⚠️ Jebakan wajib-hindari
1. **Ghost label di scene padat:** oklusi region besar (Random-Erasing/Cutout/CoarseDropout lubang besar) bisa menghapus ayam utuh tapi labelnya tetap → label noise → **menurunkan** skor. Mitigasi: lubang kecil, fraksi rendah, prioritaskan `copy_paste`/`mosaic`.
2. **`erasing` salah sasaran** (classification-only).
3. **Over-augment → underfit:** naikkan bertahap, pakai `close_mosaic`.
4. **Klaim angka lintas-domain:** gain oklusi dari ImageNet/COCO/urban ≠ gain ayam → "ekspektasi orde besaran, divalidasi empiris" (konsisten §8).

## 14. Desain ablation, ekspektasi & langkah eksekusi

### 14.1 Ablation terkontrol (ubah HANYA augmentasi; sisanya fixed: arsitektur/epoch/lr/split/seed)
| Run | Recipe | Tujuan |
|---|---|---|
| **A — baseline** | retrain yolo8m PIO, augmentasi default @ imgsz 960 di GPU | titik nol jujur (reproduksi baseline di 5090) |
| **B — +oklusi** | A + knob oklusi §13.1 | isolasi **efek oklusi** |
| **C — DG penuh** | B + HSV/photometric agresif + blur/noise + multi-scale | batas atas augmentasi bebas-target |
| (opsional **D**) | C + NP/SDG-YOLOv8 feature-perturb (custom code) | eksperimen stretch |

### 14.2 Metrik & pelaporan
Pakai **protokol fair §7 tanpa perubahan**: `n_img, n_pred, n_GT, P, R, F1, mAP@0.3/0.5/0.75/0.5:0.95`, **TIDE cls-vs-loc** (FUM), rasio pred:GT, overlay GT/pred — dievaluasi di **PIO-val + broiler + FUM + nestler** untuk tiap run. **Metrik keputusan = Δ generalisasi (B/C vs A) per dataset**, bukan mAP in-domain saja.

### 14.3 Ekspektasi & kriteria sukses (jujur)
- **Sukses realistis:** broiler ↑ tipis; FUM recall ↑ (mAP50 tetap dibatasi anotasi, dilengkapi SAHI+kalibrasi); nestler ~tetap. Gain DG single-digit = wajar.
- **Sukses ilmiah:** ablation jujur *di sumbu mana* oklusi membantu = temuan defensible + isi gap literatur.
- **Gagal-yang-bermakna:** B ≤ A di semua dataset → laporkan (kemungkinan ghost-label/over-augment) → tetap kontribusi.

### 14.4 Langkah riset+verifikasi oklusi (PLANNED — belum dijalankan)
Sebelum mengunci angka/sitasi §12–§13, jalankan **riset terverifikasi 3-vote** (5 agen: taksonomi / detection-vs-classification / crowded-poultry / domain-generalization / konfigurasi Ultralytics+5090). Tiap klaim kuantitatif diverifikasi 3 skeptik refute-oriented → hanya `CONFIRMED (≥2, 0 refute)` yang boleh dikutip. **Script workflow sudah disiapkan; jalankan setelah revisi ini disetujui.**

### 14.5 Pertanyaan terbuka (tambahan untuk §9)
1. Apakah `copy_paste` Ultralytics jalan di PIO **bbox murni** tanpa mask?
2. Ukuran & fraksi lubang CoarseDropout agar oklusi **parsial**, bukan menghapus ayam?
3. Epoch & `close_mosaic` cukup untuk konvergensi augmentasi berat di 5090?
4. Pembimbing setuju **oklusi = salah satu knob DG** (bukan penutup gap FUM/nestler)?
5. Split PIO train/val di PC 5090 identik dengan baseline? (reproducibility)

### 14.6 Checklist eksekusi (menggantikan urutan §10 untuk jalur retrain)
- [ ] **§14.4**: riset+verifikasi oklusi → isi citation library & angka `CONFIRMED`.
- [ ] **§13.3**: setup torch cu128 di PC RTX 5090; verifikasi device capability.
- [ ] **Run A** baseline retrain PIO @960 di GPU → reproduksi §1.1.
- [ ] **Run B** (+oklusi) & **Run C** (DG penuh).
- [ ] Evaluasi 4 run × 4 dataset dengan protokol §14.2 + overlay.
- [ ] Tabel Δ-generalisasi + narasi §14.3; framing nestler/FUM sesuai §12.
- [ ] (Opsional) tumpuk SAHI/TTA/kalibrasi (Tier 1) di atas model terbaik.
