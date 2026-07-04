# CHECKPOINT: Generalizability Testing — Pipeline Estimasi Berat Broiler

**Tanggal mulai**: 2026-06-27
**Status terakhir**: FASE 2 — Multi-Dataset Relative Generalization SELESAI
**Checkpoint terakhir**: Pipeline relative anomaly multi-dataset berhasil dibuat dan diverifikasi; PIO lama tetap berjalan; paket bimbingan/dokumen arah riset dibuat (`SUPERVISOR_BRIEFING_GENERALIZABILITY.md`, `RESEARCH_DIRECTION_GENERALIZABILITY.md`, `DATASET_GENERALIZABILITY_TABLE.md`, `PHASE_2_RELATIVE_GENERALIZATION.md`)

### HASIL NESTLER (2026-06-27)
- Source: Zenodo `10.5281/zenodo.20924893`
- Input: `data/external/nestler_poultry_behaviour.zip` (1.18 GB)
- Schema: `frames[*].tracks_bbox = [x1, y1, x2, y2, track_id, assembly_id]`
- Conversion: sampled 80 annotated frames per video × 6 videos
- Output dataset: `data/external/nestler_yolo/`
- Format: YOLO bbox, single class `chicken`
- Validation:
  - images = 480
  - labels = 480
  - paired = 480
  - valid bbox = 4,043
  - invalid bbox = 0
  - classes = `{0: chicken}`
  - bbox/image mean = 8.42
  - bbox/image median = 9
  - max bbox/image = 16
  - resolution = 1920×1080
  - readiness: YOLO ✅, single class ✅, uniform resolution ✅, high-density ❌
- Reports:
  - `reports/external/nestler_yolo_audit.json`
  - `reports/external/nestler_yolo_image_stats.csv`

### HASIL ROBOFLOW (2026-06-27)
API key user dipakai untuk download. Catatan keamanan: API key sudah muncul di chat/log; sebaiknya rotate/regenerate di Roboflow setelah selesai.

Dataset yang berhasil didownload dan divalidasi:

| Dataset | Images | Valid bbox | Classes | Median bbox/img | High-density | Catatan |
|---|---:|---:|---:|---:|---|---|
| `broiler_healthy_sick` | 505 | 491 | 2 | 1.0 | ❌ | Awalnya YOLO-seg; sudah dikonversi seg→bbox. 2 kelas healthy/sick. Cocok untuk classification/detection, kurang cocok untuk density. |
| `broiler_instance_seg` | 200 | 10,570 | 1 | 53.0 | ✅ | Awalnya YOLO-seg; sudah dikonversi seg→bbox. Broiler spesifik dan cukup dense. |
| `chicken_count` | 178 | 3,646 | 1 | 18.0 | ❌/borderline | YOLO bbox valid; banyak resolusi berbeda. Cocok untuk counting/domain shift. |
| `chicken_detection_fum` | 326 | 29,355 | 1 | 88.5 | ✅ | YOLO bbox valid; sangat relevan untuk dense chicken detection. Banyak resolusi berbeda. |

Dataset yang gagal/kosong:
- `broiler_detection_innodatatics`: folder terbuat tapi images=0 labels=0. Kemungkinan workspace/project/version salah atau akses tidak tersedia. Perlu cek URL Roboflow manual jika tetap ingin dipakai.

Consolidated summary:
- `reports/external/external_dataset_summary.csv`

Roboflow audit reports:
- `reports/external/broiler_healthy_sick_audit.json`
- `reports/external/broiler_instance_seg_audit.json`
- `reports/external/chicken_count_audit.json`
- `reports/external/chicken_detection_fum_audit.json`

### KEPUTUSAN STRATEGI (2026-06-27)
1. **Download paralel**: NESTLER (Zenodo, tanpa login) di background + Roboflow (perlu API key user)
2. **Mode anomaly-relatif**: Dataset tanpa info umur dijalankan TANPA berat absolut Cobb500.
   Fokus ke koreksi kamera (radial+depth) + anomaly relatif per-image. Ini jadi mode default
   untuk generalizability lintas dataset. Cobb500 absolut hanya untuk dataset yang punya week info.

---

## Latar Belakang

Dosen meminta arah riset untuk menguji apakah pipeline estimasi berat ayam broiler
(MASSA AYAM) bisa di-generalisasi ke dataset lain selain PIO. Tujuannya:

1. Menunjukkan bahwa model/pipeline ini **bukan overfitting** ke satu dataset saja
2. Membuktikan pipeline bisa bekerja di **berbagai kondisi** (lighting, density, environment)
3. Memperkuat **kontribusi riset** dengan evaluasi cross-dataset
4. Mengatasi masalah koresponden paper PIO yang tidak merespons permintaan dataset asli

### Pipeline Saat Ini (baseline di PIO dataset)

```
Dataset: PIO — 1,487 gambar, 327,289 instances, format YOLO bounding box
Standar berat: Cobb500 As-Hatched
Fitur utama: minor_axis + ellipse_area + age/week
Koreksi kamera: radial_depth_median
Anomaly: percentile method P97/P99
```

Hasil baseline PIO:
- Total bbox valid: 321,427
- Percentile P99 critical anomaly: 3,667 bbox (1.14%)
- Image-level abnormal: 617 gambar (42.03%)

---

## RENCANA FASE-FASE

### FASE 1: Identifikasi & Pengumpulan Dataset [IN PROGRESS]

#### 1.1 Analisis Table 1 Paper PIO [DONE]

Dataset dari Table 1 "Comparative summary of relevant broiler chicken detection datasets":

| # | Dataset | Ref | Annotation | Environment | Density | Publik? | Prioritas |
|---|---------|-----|-----------|-------------|---------|---------|-----------|
| 1 | Zhuang & Zhang 2019 | [3] | BBox | Commercial (China), partially controlled | No | ❓ Perlu cek | SEDANG |
| 2 | ChickTrack | [11] | BBox (Tracking) | Realistic farm videos | Yes | ❓ Perlu cek | TINGGI |
| 3 | Van Der Eijk et al. 2022 | [6] | Segmentation | Resource monitoring | No | ❓ Perlu cek | RENDAH (segmentation → perlu konversi) |
| 4 | Guo et al. 2022 | [5] | BBox / Classification | Controlled experimental | No | ❓ Perlu cek | SEDANG |
| 5 | Dense-Chicken | [7] | Density maps | High-density flock | Yes | ❓ Perlu cek | RENDAH (density maps, bukan bbox) |
| 6 | Chicks4FreeID | [8] | Semantic segmentation | Controlled, neutral background | No | ❓ Perlu cek | RENDAH (segmentation + background beda drastis) |
| 7 | Broiler-Net | [4] | BBox | Internet-sourced videos | Yes | ❓ Perlu cek | SEDANG |
| 8 | Qi et al. 2025 | [9] | BBox (Tracking) | 20-day video study | No | ❓ Perlu cek | SEDANG |
| 9 | Yang et al. 2026 | [10] | Multimodal (Visible/Thermal) | Caged hen houses | Yes | ❓ Perlu cek | RENDAH (laying hens, bukan broiler) |
| 10 | PIO (present work) | [13] | BBox | Real commercial farm | Yes | ✅ Zenodo | BASELINE (sudah dipakai) |

Kriteria pemilihan dataset:
- **WAJIB**: Punya bounding box annotation (atau bisa dikonversi)
- **WAJIB**: Publik & bisa didownload
- **DIUTAMAKAN**: Broiler chicken (bukan laying hen)
- **DIUTAMAKAN**: Overhead/top-view camera (mirip setup PIO)
- **DIUTAMAKAN**: Punya informasi umur/growth stage

#### 1.2 Pencarian Dataset Publik Tambahan [DONE]

Sumber yang sudah dicek: Zenodo, Roboflow Universe, GitHub, Hugging Face, ScienceDirect, survey paper.

**HASIL — Status ketersediaan dataset dari Table 1:**

| Dataset Table 1 | Status publik | Catatan |
|-----------------|--------------|---------|
| Zhuang & Zhang [3] | ❌ Tidak ditemukan publik | Dataset China, on-request |
| ChickTrack [11] | ❌ Tidak publik | "our dataset", tidak ada repo |
| Van Der Eijk [6] | ❓ Belum dicek detail | Segmentation |
| Guo et al. [5] | ❓ Belum dicek detail | - |
| Dense-Chicken [7] | ❌ Tidak ditemukan publik | Density maps |
| Chicks4FreeID [8] | ⚠️ Publik TAPI tidak cocok | HuggingFace `dariakern/Chicks4FreeID` — hanya cropped images + identity label, **tidak ada bbox**, fokus re-ID |
| Broiler-Net [4] | ⚠️ Paper di arXiv | Pakai internet videos, dataset tidak jelas dirilis |
| Qi et al. [9] | ❓ Belum dicek detail | - |
| Yang et al. [10] | ❌ Laying hen, bukan broiler | Multimodal thermal |

**KESIMPULAN PENTING**: Mayoritas dataset Table 1 TIDAK tersedia publik dengan bbox.
Strategi diganti → pakai dataset publik siap-YOLO dari **Roboflow Universe** + **Zenodo**
yang fungsinya setara (broiler detection, bounding box).

---

#### 1.2b Dataset Publik yang DIPILIH untuk Generalizability Testing [DONE]

**TIER 1 — Siap pakai (YOLO bbox, langsung downloadable):**

| # | Dataset | Sumber | ~Images | Format | Catatan |
|---|---------|--------|---------|--------|---------|
| A | PIO (baseline) | Zenodo `16686320` | 1,487 | YOLO bbox | ✅ Sudah ada di `data/`. Commercial+prototype, overhead, ada week info |
| B | Broiler Chicken Detection | Roboflow `innodatatics` | 179 | YOLO bbox | Broiler spesifik |
| C | Broilerbird | Roboflow | 309 | YOLO bbox | Broiler |
| D | broiler_chicks_v2 | Roboflow | 165 | YOLO bbox | Broiler, anak ayam |
| E | Broiler Healthy & Sick | Roboflow `technicalresearch` | 209 | YOLO bbox | 2 kelas (sehat/sakit) |
| F | Chicken Detection | Roboflow `fum-icce` | 157 | YOLO bbox | General chicken |
| G | chicken count | Roboflow `chickendetection` | 100 | YOLO bbox | Counting, dense |

**TIER 2 — Perlu konversi:**

| # | Dataset | Sumber | Format | Konversi |
|---|---------|--------|--------|----------|
| H | NESTLER Poultry Behaviour | Zenodo `20924893` | bbox + keypoints (video) | Parse → YOLO bbox |
| I | Broiler Instance Segmentation | Roboflow `broiler-data` | segmentation mask | seg → bbox |

**TIDAK DIPAKAI:**
- Chicks4FreeID (cropped, re-ID, no bbox)
- Shams 2025 weight estimation (on-request, tidak publik) — TAPI relevan sebagai referensi metode
- ChickTrack, Dense-Chicken, Zhuang (tidak publik)

> **Cara download Roboflow** (perlu API key gratis dari roboflow.com):
> ```python
> import roboflow
> rf = roboflow.Roboflow(api_key="YOUR_KEY")
> project = rf.workspace("WORKSPACE").project("PROJECT")
> dataset = project.version(N).download("yolov8")
> ```

#### 1.3 Download & Validasi Dataset [TODO — LANGKAH BERIKUTNYA]

Untuk setiap dataset Tier 1:
- [ ] Download dataset (Roboflow butuh API key user, atau download manual via browser)
- [ ] Verifikasi format annotation (konversi ke YOLO jika perlu)
- [ ] Hitung statistik: jumlah gambar, jumlah bbox, distribusi ukuran
- [ ] Cek apakah ada informasi umur/growth stage (kemungkinan TIDAK ada di Roboflow)
- [ ] Simpan di `data/external/<nama_dataset>/`

> **CATATAN**: Roboflow datasets kemungkinan besar **tidak punya info umur/week**.
> Ini berarti FASE 2.3 (handling missing age) jadi krusial. Pipeline harus bisa jalan
> mode "anomaly relatif per-image" tanpa estimasi berat absolut Cobb500.

---

### FASE 2: Adaptasi Pipeline untuk Multi-Dataset [TODO]

#### 2.1 Refactor Pipeline [TODO]

- [ ] Modifikasi `scripts/common.py` untuk mendukung multiple dataset configs
- [ ] Buat `configs/datasets/` folder dengan config per dataset
- [ ] Setiap config berisi:
  - path ke images & labels
  - mapping umur (jika tersedia)
  - resolusi gambar
  - standar berat yang dipakai
  - metadata khusus dataset

#### 2.2 Konversi Annotation [TODO]

- [ ] Script konversi VOC XML → YOLO txt (jika perlu)
- [ ] Script konversi COCO JSON → YOLO txt (jika perlu)
- [ ] Script konversi segmentation mask → bounding box (jika perlu)
- [ ] Validasi konversi: overlay bbox ke gambar untuk visual check

#### 2.3 Handling Missing Age/Week Info [TODO]

Beberapa dataset mungkin tidak punya info umur. Strategi:
- **Opsi A**: Skip estimasi berat absolut, fokus ke anomaly relatif per-image saja
- **Opsi B**: Estimasi umur dari rata-rata ukuran bbox (heuristic)
- **Opsi C**: Jalankan pipeline per-subset jika dataset punya partisi temporal

---

### FASE 3: Eksekusi Pipeline per Dataset [TODO]

Untuk setiap dataset:
1. [ ] `audit_dataset.py` — audit kualitas data
2. [ ] `extract_bbox_features.py` — ekstrak fitur bbox
3. [ ] `estimate_weight_anomalies.py` — estimasi berat
4. [ ] `compare_camera_corrections.py` — bandingkan koreksi kamera
5. [ ] `image_level_anomaly.py` — anomaly per gambar

#### 3.1 Metrics yang Dikumpulkan per Dataset [TODO]

```
- total_images
- total_valid_bbox
- mean_bbox_per_image
- radial_correction_improvement_%
- depth_correction_improvement_%
- image_abnormal_rate_%
- percentile_P97_candidate_%
- percentile_P99_critical_%
- cross_validation_consistency
```

---

### FASE 4: Analisis Cross-Dataset & Generalizability [TODO]

#### 4.1 Tabel Perbandingan [TODO]

Buat tabel perbandingan metrik antar dataset:
- Apakah pola koreksi radial konsisten?
- Apakah threshold anomaly P97/P99 stabil?
- Apakah distribusi anomaly per week/stage mirip?

#### 4.2 Cross-Dataset Validation [TODO]

- [ ] Train/calibrate pada PIO → test pada dataset lain
- [ ] Train/calibrate pada dataset lain → test pada PIO
- [ ] Leave-one-dataset-out evaluation

#### 4.3 Analisis Domain Shift [TODO]

Faktor yang mempengaruhi generalizability:
- Perbedaan resolusi kamera
- Perbedaan sudut pandang (overhead vs angled)
- Perbedaan jenis kandang (commercial vs prototype vs controlled)
- Perbedaan strain ayam (jika diketahui)
- Perbedaan lighting (natural vs artificial)
- Perbedaan density (crowded vs sparse)

---

### FASE 5: Laporan & Dokumentasi [TODO]

#### 5.1 Output Final [TODO]

```
reports/cross_dataset_comparison.html
reports/cross_dataset_comparison.csv
reports/generalizability_summary.json
reports/domain_shift_analysis.html
```

#### 5.2 Kontribusi Riset [TODO]

Narasi yang bisa dimasukkan ke paper:
1. Pipeline estimasi berat berbasis 2D feature + Cobb500 bersifat **generalizable/tidak**
2. Koreksi radial+depth konsisten/inkonsisten antar dataset
3. Percentile anomaly threshold stabil/perlu adaptasi per dataset
4. Rekomendasi untuk deployment di berbagai kondisi kandang

---

## STATUS CHECKPOINT

| Fase | Status | Terakhir dikerjakan |
|------|--------|-------------------|
| 1.1 Analisis Table 1 | ✅ DONE | 2026-06-27 |
| 1.2 Pencarian dataset publik | ✅ DONE | 2026-06-27 |
| 1.2b Pemilihan dataset | ✅ DONE | 2026-06-27 |
| 1.3 Download & validasi | ✅ DONE | PIO + NESTLER + 4 Roboflow tervalidasi; 1 kandidat Roboflow kosong/gagal |
| 2.1 Refactor pipeline | ✅ DONE | Pipeline eksternal baru dibuat tanpa merusak PIO lama |
| 2.2 Konversi annotation | ✅ DONE | Roboflow seg→bbox dan NESTLER video→YOLO selesai di Fase 1.3 |
| 2.3 Handling missing info | ✅ DONE | Mode anomaly-relatif tanpa Cobb500 dibuat dan diverifikasi |
| 3 Eksekusi per dataset | ✅ DONE | 5 dataset eksternal diproses via `run_external_relative_pipeline.py` |
| 4 Analisis cross-dataset | ✅ DONE | `cross_dataset_relative_summary.*` dan HTML report dibuat |
| 5 Laporan & dokumentasi | ⏳ TODO | - |

---

## CATATAN PENTING

### Dataset yang Sudah Dikonfirmasi
- **PIO**: ✅ Sudah ada di `data/` — 1,035 train + 452 val images, YOLO format
  - Zenodo: https://doi.org/10.5281/zenodo.16686320

### Dataset yang Dipilih (hasil pencarian 2026-06-27)
- **Tier 1 (siap YOLO bbox)**: PIO ✅, + 6 dataset Roboflow (Broiler Detection, Broilerbird, broiler_chicks_v2, Healthy&Sick, FUM Chicken Detection, chicken count)
- **Tier 2 (perlu konversi)**: NESTLER (Zenodo), Broiler Instance Segmentation (Roboflow)
- Detail lengkap di Fase 1.2b di atas

### Hambatan yang Diketahui
1. Koresponden paper PIO tidak merespons → dataset utama tetap yang ada di Zenodo ✅
2. **KONFIRMASI: Mayoritas dataset Table 1 TIDAK publik** → diganti dataset Roboflow setara
3. Dataset Roboflow kemungkinan **tidak punya info umur/week** → pipeline harus mode anomaly-relatif
4. Perbedaan strain ayam (Cobb500 vs Ross 308 vs lokal) mempengaruhi standar berat absolut
5. Download Roboflow butuh **API key gratis** dari user (perlu daftar di roboflow.com)
6. Resolusi & sudut kamera dataset Roboflow bervariasi → uji domain shift jadi penting

### Instruksi untuk Melanjutkan
Jika token habis, lanjutkan dari checkpoint terakhir:
1. Buka `CHECKPOINT.md` ini
2. Cek kolom "Status" di tabel di atas
3. Lanjutkan dari fase yang belum selesai
4. Update status dan tanggal setiap kali ada progress
