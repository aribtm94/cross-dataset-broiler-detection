# Riset Referensi: Pivot ke Pipeline End-to-End (Multi-Model ML + Depth Estimation)

Tanggal: 2026-07-01
Status: hasil deep-research (belum diimplementasikan ke kode)
Terkait: [[RESEARCH_DIRECTION_GENERALIZABILITY.md]], [[REVIEW_METODOLOGI_GENERALISASI.md]]

---

## 0. Latar Belakang Pivot

Pipeline saat ini bersifat **fully algorithmic/heuristik**: fitur bbox YOLO (minor axis, ellipse
area) → koreksi kamera radial + depth-light (`bottom_y_norm` sebagai proxy depth kasar) → kalibrasi
Cobb500 → threshold percentile P97/P99. Review metodologis (`REVIEW_METODOLOGI_GENERALISASI.md`,
poin **C1**) sudah menandai kelemahan fundamental: **tidak ada model/prediksi terlatih yang diuji** —
yang dievaluasi adalah heuristik di atas label ground-truth YOLO, bukan performa model.

Arah baru yang diminta: pivot ke pipeline **end-to-end** yang mengombinasikan beberapa model ML
terlatih (deteksi + **estimasi depth** + regresi berat/ukuran), diterapkan pada **3 dataset lokal**
di `data/data/external/`:

| Dataset lokal | Folder | Karakteristik |
|---|---|---|
| Nestler | `nestler_yolo` | sparse, hasil sampling video, frame berkorelasi |
| Broiler Instance Segmentation | `broiler_instance_seg` | dense |
| FUM Chicken Detection | `chicken_detection_fum` | dense, high-density |

Ketiganya **tidak punya ground-truth berat/umur** — sama seperti kondisi yang sudah dibahas di
`RESEARCH_DIRECTION_GENERALIZABILITY.md`.

Dokumen ini merangkum hasil deep-research (109 sub-agent, 26 sumber di-fetch, 117 klaim diekstrak,
25 klaim diverifikasi adversarial 3-vote → 14 confirmed / 11 refuted) untuk empat area:

1. Metode end-to-end livestock/poultry weight estimation.
2. Model monocular depth estimation yang cocok untuk kamera top-down/fisheye peternakan.
3. Teknik training tanpa ground-truth berat (weak supervision).
4. Preseden generalisasi lintas-dataset/kandang di agricultural CV.

**Catatan penting:** tidak ada satu pun paper yang menunjukkan kombinasi persis
"poultry + depth + no-weight-labels". Semua preseden livestock end-to-end datang dari **sapi, babi,
sapi perah, atau bebek** — bukan ayam broiler. Kombinasi yang direkomendasikan di bawah adalah
**sintesis baru**, bukan metode yang sudah pernah dibuktikan persis seperti ini. Ini harus dinyatakan
eksplisit di skripsi sebagai **kontribusi/novelty**, bukan diklaim sebagai kombinasi yang sudah mapan.

---

## 1. Referensi Terverifikasi per Area

### 1.1 Deteksi/segmentasi + regresi berat (livestock end-to-end)

| # | Klaim | Sumber | Confidence |
|---|---|---|---|
| 1 | Preseden sapi: segmentasi (Mask R-CNN *atau* Xception+Grad-CAM/Puzzle-CAM weakly-supervised dari label image-level "Cattle vs Background", **tanpa mask piksel**) → ekstrak 12 fitur geometris (area, panjang, lebar tubuh) → regresi berat via DNN dengan residual/shortcut connection, pooling, dropout, dense layers. | [Lee, Lee, Cho 2023 — MDPI Applied Sciences](https://www.mdpi.com/2076-3417/13/5/2896) | High (arsitektur); **angka akurasi MAE/MAPE full-vs-weak DITOLAK verifikasi — jangan dikutip** |
| 2 | Segmentasi instance weakly-supervised **hanya dari bounding box** (tanpa mask) untuk babi, mengadaptasi framework BoxTeacher — langsung applicable ke pipeline berbasis dataset YOLO bbox tanpa mask/berat ground-truth. | [Zhou et al. 2025 — Nature Sci. Reports via PubMed](https://pubmed.ncbi.nlm.nih.gov/40467907/) | High (metodologi box-only); **klaim kuantitatif "gap 3% vs fully-supervised, kalahkan 3 dari 5 baseline" DITOLAK verifikasi** |
| 3 | Preseden bebek: fusi RGB top-view + RGB side-view + depth image + point cloud 3D (ResNet50 branches + PointNet++ + Transformer fusion) → regresi 8 target (berat + 7 dimensi tubuh), MAPE 6.33%, R²=0.953. | [arXiv 2503.14001](https://arxiv.org/html/2503.14001v1) — dipublikasi di *Agriculture* MDPI 2025 | Medium (angka MAPE/R² confirmed; **deskripsi arsitektur persis tiga-branch DITOLAK verifikasi**, treat sebagai perkiraan) |
| 4 | MFF-GBDT (bukan dari verify-pass, disebut di fetch log): Mask R-CNN + 25 fitur geometris 2D/3D + 2048 fitur deep dari ResNet50 kustom → fusion → gradient boosting (LightGBM/XGBoost) untuk prediksi berat broiler. | fetch log (belum lolos verify-pass eksplisit) | Perlu dicek ulang manual sebelum dikutip |

### 1.2 Model monocular depth estimation

| # | Klaim | Sumber | Confidence |
|---|---|---|---|
| 5 | **Calibration Tokens (ICCV 2025)** — token embedding kecil yang ditambahkan ke FMDE (foundational monocular depth estimator) terlatih seperti MiDaS, Depth Anything ViT-L, UniDepth ViT-S, agar bisa dipakai pada gambar fisheye **tanpa retraining base model dan tanpa intrinsic kamera saat inference**. Satu set token generalize ke berbagai parameter distorsi & domain indoor/outdoor. Training **self-supervised**: pseudo-label dari model depth itu sendiri pada dataset perspective publik, gambar fisheye sintetis via model distorsi Kannala-Brandt, hanya perlu ~200K sampel. Token bisa dilepas untuk kembali ke performa original pada gambar perspective. | [arXiv 2508.04928](https://arxiv.org/html/2508.04928v1) | **High** — satu-satunya metode adaptasi fisheye yang lolos verifikasi bersih |
| 6 | **Depth Anything V2** — training 3 tahap (teacher pada gambar sintetis → pseudo-label 62M gambar real → student lebih kecil dilatih hanya pada pseudo-label, sengaja hindari label depth real yang noisy). Tersedia skala 25M–1.3B parameter, varian terkecil real-time. Pada benchmark wildlife camera-trap (outdoor, non-poultry) meraih akurasi metric-depth zero-shot terbaik di antara 4 model (MAE 0.454m, korelasi 0.962), mengalahkan ZoeDepth (MAE 3.087m), ML Depth Pro (MAE 1.127m), Metric3D v2 (MAE 0.867m/korelasi 0.974). Penulis benchmark merekomendasikan Depth Anything V2 + ekstraksi depth median per-region. | [arXiv 2406.09414](https://arxiv.org/html/2406.09414v1) (paper asli, NeurIPS 2024), [arXiv 2510.04723](https://arxiv.org/html/2510.04723v1) (benchmark independen, Univ. Firenze, Okt 2025) | **High** |
| 7 | **UniDepth** — prediksi point cloud 3D metric langsung dari satu gambar tanpa perlu intrinsic kamera saat inference, pakai representasi pseudo-spherical yang memisahkan geometri kamera dari depth. Didesain untuk generalisasi lintas-kamera. **Catatan**: masih butuh data kamera terkalibrasi saat *training*, dan range depth dibatasi oleh sensor training — bukan solusi calibration-free penuh. | [arXiv 2501.11841](https://arxiv.org/html/2501.11841v3) | Medium (desain arsitektur confirmed; klaim generalisasi cross-camera yang kuat DITOLAK) |
| 8 | **ZoeDepth**, meski model "metric depth", akurasinya **turun tajam** di luar domain training-nya (indoor/driving) — peringatan agar tidak asumsi model metric-depth otomatis generalize lintas domain. | [arXiv 2510.04723](https://arxiv.org/html/2510.04723v1) | High |

**Ditolak verifikasi (jangan dikutip tanpa cek ulang sumber primer):**
- **Depth Any Camera (DAC)** — klaim extend depth estimator ke fisheye/360° tanpa data task-specific: **ditolak (0-3)**.
- **Metric3D / Metric3D v2** — klaim kalibrasi-free cross-camera generalization, klaim training pada 16 juta gambar lintas ribuan kamera, klaim akurat untuk single-image metrology: **semua ditolak (0-3 / 1-2)**.

### 1.3 Weak supervision / tanpa ground-truth berat

| # | Klaim | Sumber | Confidence |
|---|---|---|---|
| 9 | Segmentasi weakly-supervised (label image-level saja, tanpa mask piksel) adalah paradigma training yang valid di livestock CV (lihat #1, #2 di atas) — **hanya fakta kelayakan metodologi yang confirmed, bukan angka akurasi spesifik**. | (lihat #1, #2) | High (metodologi), rendah untuk angka |
| 10 | Transfer learning lintas kandang/farm untuk prediksi berat dari depth image & point cloud sudah diteliti di sapi perah: membandingkan single-source, joint, dan transfer-learning di 4 arsitektur (ConvNeXt, MobileViT untuk depth image; PointNet, DGCNN untuk point cloud) di 3 ukuran farm — template desain eksperimen untuk menangani domain shift. **Hasil "transfer > single-source > joint" DITOLAK verifikasi** — hanya desain studinya yang reliable. | [arXiv 2601.01044](https://arxiv.org/pdf/2601.01044) | Medium |

### 1.4 Domain generalization lintas dataset/kandang

Area ini paling lemah: klaim survey domain-adaptation agricultural CV (`arXiv 2506.05972`) tentang
degradasi akurasi lintas kandang dan analogi cattle re-identification **keduanya ditolak verifikasi**.
Tidak ada temuan area ini yang lolos verifikasi bersih — argumen soal risiko domain shift harus
dibangun dari prinsip dasar / bukti lain di dokumen ini (mis. ZoeDepth yang gagal cross-domain, #8),
bukan dikutip dari survey tersebut.

---

## 2. Rekomendasi Arsitektur

### Opsi A (direkomendasikan) — YOLO existing + Depth Anything V2 + regresi fusion

```text
1. Deteksi:   YOLO yang sudah dilatih di PIO (tetap dipakai, tidak perlu model baru)
2. Depth:     Depth Anything V2 (ViT-S/ViT-B, 25M-100M param, ringan, real-time)
              -> ekstrak depth median per-bbox (bukan mean; ikuti rekomendasi benchmark #6)
              -> menggantikan proxy bottom_y_norm yang sekarang dipakai
3. Opsional:  Jika salah satu dari 3 dataset lokal pakai lensa fisheye/wide-angle,
              bungkus Depth Anything V2 dengan Calibration Tokens (#5)
              -> tidak perlu retrain backbone, tidak perlu intrinsic kamera saat inference
4. Fusion:    MLP/DNN kecil dengan residual connection (pola arsitektur sapi, #1)
              input: ellipse_area, minor_axis (fitur geometris existing)
                   + depth median per-bbox (scale-normalized)
              output: estimasi berat/ukuran
5. Supervisi: TIDAK ADA ground-truth berat di 3 dataset lokal -> pakai kurva pertumbuhan
              Cobb500 (age-conditioned) sebagai weak/pseudo-label untuk regresi,
              analog dengan cara Depth Anything V2 sendiri dilatih dari pseudo-label
              teacher model, dan cara paper sapi/babi (#1, #2) mengganti dense
              ground-truth dengan supervisi box/image-level.
```

Alasan pemilihan:
- Compute rendah (semua backbone punya varian kecil/frozen) — cocok skala skripsi.
- Tidak butuh data timbang manual.
- Tidak butuh retrain YOLO yang sudah ada.
- Evaluasi dilakukan lewat **konsistensi depth** dan **cross-dataset relative-ranking**
  (bukan MAE absolut, karena ground-truth tidak tersedia) — selaras dengan kritik C1/C2 di
  `REVIEW_METODOLOGI_GENERALISASI.md` yang menuntut metrik yang *bisa gagal*.

### Opsi B — Weakly-supervised segmentation + fitur geometris + GBDT

```text
Deteksi bbox (existing YOLO) -> box-only weakly-supervised segmentation (pola BoxTeacher, #2)
-> ekstrak fitur geometris 2D (area, panjang, lebar) dari mask hasil segmentasi
-> gradient boosting (LightGBM/XGBoost) untuk regresi berat/ukuran, pola MFF-GBDT
```
Lebih murah dari Opsi A (tidak perlu model depth), tapi tidak memenuhi permintaan eksplisit
"salah satu metode pakai depth estimation" dari pembimbing/mahasiswa.

### Opsi C — Full multimodal fusion (pola bebek, #3)

```text
RGB top-view + depth image + fitur 2D -> multiple CNN branch -> fusion (attention/transformer)
-> regresi multi-target (berat + ukuran tubuh)
```
Paling mendekati riset preseden (#3), tapi compute lebih berat dan butuh desain fusion custom —
risiko tinggi untuk skala skripsi dengan deadline terbatas, dan angka akurasi persis paper ini
tidak lolos verifikasi arsitektur (hanya metrik akhir yang confirmed).

**Rekomendasi:** mulai dari **Opsi A**. Opsi B sebagai fallback murah bila depth model terbukti
tidak stabil pada salah satu dataset. Opsi C sebagai extension kalau waktu memungkinkan.

---

## 3. Klaim yang Aman vs Tidak Boleh Dibuat

Boleh diklaim:

1. Pipeline dipivot dari heuristik murni menjadi kombinasi model terlatih (deteksi + depth +
   regresi fusion) — mengatasi kritik C1 (tidak ada model/prediksi yang diuji).
2. Depth Anything V2 dan Calibration Tokens adalah pilihan depth estimator yang didukung
   literatur 2025 untuk skenario ringan/tanpa kalibrasi kamera spesifik.
3. Pola arsitektur "segment/detect → fitur geometris + depth → regresi DNN residual" punya
   preseden di livestock CV lain (sapi, babi, bebek), meski belum pernah diuji khusus di
   broiler/poultry dengan kamera top-down.
4. Karena tidak ada ground-truth berat di 3 dataset lokal, supervisi memakai Cobb500 sebagai
   weak/pseudo-label — pola ini konsisten dengan cara model depth sendiri (Depth Anything V2)
   dan paper livestock lain mengatasi ketiadaan label dense.

Tidak boleh diklaim:

1. Kombinasi "poultry + depth estimation + no-weight-labels" ini sudah pernah dibuktikan di
   paper manapun — ini **gap literatur**, kombinasinya adalah kontribusi baru skripsi ini.
2. Angka akurasi spesifik dari paper sapi (#1) atau babi (#2) — angka-angka itu **ditolak**
   verifikasi adversarial dan tidak boleh dikutip sebagai ekspektasi performa.
3. Model depth manapun (termasuk Depth Anything V2) "calibration-free" secara mutlak di semua
   kondisi kamera — hanya Calibration Tokens yang punya klaim calibration-free yang lolos
   verifikasi, dan itu pun spesifik untuk domain fisheye yang diujikan di papernya.
4. DAC atau Metric3D sebagai solusi cross-camera zero-shot — klaim-klaim itu **ditolak**
   verifikasi dan tidak boleh dipakai sebagai dasar pemilihan model.

---

## 4. Pertanyaan Terbuka (perlu dijawab sebelum implementasi)

1. Apakah ada paper yang menerapkan monocular depth estimation khusus untuk estimasi
   berat/ukuran broiler (langsung maupun tidak langsung)? Berdasarkan riset ini: **tidak
   ditemukan** — kemungkinan ini benar-benar gap yang belum digarap.
2. Karena 3 dataset eksternal tidak punya ground-truth berat, metrik evaluasi apa yang bisa
   menggantikan MAE/mAP agar meyakinkan penguji skripsi? (kandidat: depth-consistency check,
   relative-ranking plausibility terhadap kurva Cobb500, bukan akurasi absolut).
3. Apakah `nestler_yolo`, `broiler_instance_seg`, `chicken_detection_fum` memakai lensa
   fisheye/wide-angle atau kamera perspective standar? Ini menentukan apakah komponen
   Calibration Tokens benar-benar diperlukan, atau cukup Depth Anything V2 polos.
4. Berapa budget compute/GPU yang tersedia? Menentukan pilihan ViT-S/ViT-B (ringan) vs model
   metric-depth yang lebih besar (UniDepth/Metric3D v2).

---

## 5. Catatan Reliabilitas Sumber

- Beberapa sumber kunci (Calibration Tokens, benchmark wildlife depth, paper transfer-learning
  sapi perah) adalah **preprint 2025 yang sangat baru**, belum banyak disitasi/direplikasi
  independen — layak dikutip di skripsi 2026 tapi nyatakan sebagai temuan baru, bukan konsensus
  mapan.
- Semua angka benchmark depth model (Depth Anything V2, ZoeDepth, Metric3D v2, ML Depth Pro)
  berasal dari **satu preprint non-poultry** (wildlife camera-trap, n=93 gambar, Okt 2025) —
  arahnya berguna tapi sampel kecil dan domain berbeda dari kamera top-down peternakan; anggap
  angka MAE sebagai ilustratif, bukan langsung transferable.
- Sumber lengkap (26 total, kualitas primary/secondary/blog) tersedia di transcript workflow;
  daftar klaim yang **ditolak** verifikasi didokumentasikan di Bagian 1 agar tidak terulang
  dikutip secara keliru.

---

## 6. Langkah Berikutnya

1. Cek metadata/EXIF atau sampel visual dari 3 dataset lokal untuk menjawab pertanyaan terbuka
   #3 (fisheye vs perspective).
2. Prototipe cepat: jalankan Depth Anything V2 (ViT-S) pada beberapa sampel dari tiap dataset,
   cek kualitas depth map secara visual sebelum investasi penuh ke pipeline fusion.
3. Desain skema weak-label dari Cobb500 (age-conditioned growth curve) sebagai pseudo-target
   regresi — putuskan bagaimana usia/minggu diestimasi untuk dataset yang tidak,punya metadata
   umur di filename (berbeda dari PIO).
4. Tulis bagian metodologi skripsi yang menyatakan eksplisit gap literatur di Bagian 0/3 sebagai
   dasar kontribusi riset, bukan replikasi kombinasi yang sudah ada.
