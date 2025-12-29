# ENSO Forecasting Project: Technical Documentation

## Daftar Isi
1. [Pendahuluan](#1-pendahuluan)
2. [Latar Belakang Oseanografi](#2-latar-belakang-oseanografi)
3. [Arsitektur Data](#3-arsitektur-data)
4. [Pipeline Preprocessing](#4-pipeline-preprocessing)
5. [Arsitektur Model LSTM](#5-arsitektur-model-lstm)
6. [Mekanisme Multivariate Input](#6-mekanisme-multivariate-input)
7. [Strategi Validasi](#7-strategi-validasi)
8. [Interpretasi Hasil](#8-interpretasi-hasil)

---

## 1. Pendahuluan

### 1.1 Tujuan Proyek
Proyek ini bertujuan untuk memprediksi **Anomali Suhu Permukaan Laut (SST)** di perairan Indonesia menggunakan pendekatan Deep Learning. Anomali SST adalah penyimpangan suhu dari nilai klimatologi (rata-rata jangka panjang), yang mengindikasikan kondisi lebih hangat atau lebih dingin dari biasanya.

### 1.2 Mengapa Ini Penting?
- **Coral Bleaching**: Anomali positif ekstrem (>1°C) menyebabkan pemutihan karang massal
- **Perikanan**: Suhu mempengaruhi distribusi ikan dan produktivitas perairan
- **Cuaca & Iklim**: SST Indonesia berkorelasi dengan curah hujan di Asia Tenggara
- **Early Warning System**: Prediksi akurat membantu mitigasi dampak El Niño/La Niña

### 1.3 Inovasi Proyek Ini
Berbeda dengan prediksi time series biasa, model ini menggunakan pendekatan **Multivariate** dengan memasukkan **Indeks Niño 3.4** sebagai prediktor eksternal. Ini memanfaatkan fenomena **telekoneksi** dimana kondisi di Pasifik Tengah mempengaruhi Indonesia.

---

## 2. Latar Belakang Oseanografi

### 2.1 El Niño-Southern Oscillation (ENSO)
ENSO adalah fenomena iklim terbesar di bumi yang mempengaruhi cuaca global. Terdiri dari dua komponen:
- **El Niño**: Pemanasan abnormal di Pasifik Tengah/Timur
- **La Niña**: Pendinginan abnormal di Pasifik Tengah/Timur

### 2.2 Indeks Niño 3.4
Niño 3.4 adalah indeks standar untuk mengukur fase ENSO, dihitung dari anomali SST di wilayah:
- **Koordinat**: 5°N - 5°S, 170°W - 120°W (Pasifik Tengah)
- **Threshold El Niño**: > +0.5°C (selama 5 bulan berturut-turut)
- **Threshold La Niña**: < -0.5°C (selama 5 bulan berturut-turut)

### 2.3 Telekoneksi: Pasifik → Indonesia
Mekanisme bagaimana ENSO mempengaruhi Indonesia:

```
El Niño di Pasifik → Sirkulasi Walker melemah → 
→ Trade winds melemah → Indonesian Throughflow berkurang →
→ Air hangat tidak masuk ke Indonesia → SST Indonesia TURUN
```

**Time Lag**: Efek ENSO biasanya terasa di Indonesia setelah 1-3 bulan.

### 2.4 Wilayah Studi: Indonesia
Koordinat bounding box:
- **Latitude**: 11°S - 6°N
- **Longitude**: 95°E - 141°E
- Mencakup: Laut Jawa, Laut Banda, Laut Sulawesi, Laut Flores

---

## 3. Arsitektur Data

### 3.1 Sumber Data

| Data | Sumber | Resolusi | Format |
|------|--------|----------|--------|
| Indonesian SST | NOAA OISST V2 High Resolution | 0.25° × 0.25° harian | NetCDF |
| Niño 3.4 Index | NOAA ERSSTv5 | Bulanan | Text file |

### 3.2 Struktur Folder
```
enso-forecasting/
├── data/
│   ├── raw/
│   │   └── nina34.anom.data.txt    # Niño 3.4 Index (1950-2025)
│   └── processed/
│       └── sst_indo_clean.csv      # SST Indonesia bulanan (2000-2012)
├── data_sst/                        # Raw NetCDF (~500MB/file)
│   ├── sst.day.mean.2000.nc
│   ├── sst.day.mean.2001.nc
│   └── ...
└── output/
    └── figures/                     # Hasil visualisasi
```

### 3.3 Format Data Akhir
File `sst_indo_clean.csv`:
```csv
date,sst_actual,sst_anomaly
2000-01-01,28.5445,-0.1766
2000-02-01,28.3332,-0.3313
...
```

Kolom:
- `date`: Tanggal (format YYYY-MM-DD, selalu tanggal 1)
- `sst_actual`: SST rata-rata Indonesia dalam °C
- `sst_anomaly`: Penyimpangan dari klimatologi bulanan

---

## 4. Pipeline Preprocessing

### 4.1 Alur Kerja
```
NetCDF (Harian, Global) → Slice Indonesia → Resample Bulanan → 
→ Spatial Mean → Hitung Klimatologi → Hitung Anomali → CSV
```

### 4.2 Detail Setiap Langkah

#### Langkah 1: Slice Wilayah Indonesia
```python
ds_indo = ds.sel(
    lat=slice(-11, 6),
    lon=slice(95, 141)
)
```
Dari grid global 1440×720, hanya mengambil ~180×68 grid points.

#### Langkah 2: Resample ke Bulanan
```python
ds_monthly = ds.resample(time='MS').mean(dim='time')
```
Mengubah 365 titik data harian → 12 titik data bulanan.

#### Langkah 3: Spatial Mean
```python
sst_mean = ds['sst'].mean(dim=['lat', 'lon'])
```
Merata-ratakan seluruh grid Indonesia menjadi 1 nilai per bulan.

#### Langkah 4: Menghitung Klimatologi
Klimatologi = rata-rata jangka panjang untuk setiap bulan:
```python
climatology = sst_series.groupby('time.month').mean('time')
# Menghasilkan 12 nilai: Jan=28.7°C, Feb=28.6°C, ..., Dec=29.0°C
```

#### Langkah 5: Menghitung Anomali
```python
anomaly = observed_sst - climatology
# Contoh: Jika Jan 2010 = 29.2°C dan klimatologi Jan = 28.7°C
# Maka anomali = +0.5°C (lebih hangat dari normal)
```

---

## 5. Arsitektur Model LSTM

### 5.1 Mengapa LSTM?
Long Short-Term Memory (LSTM) dipilih karena:
- Mampu mengingat dependensi jangka panjang (12+ bulan)
- Mengatasi masalah vanishing gradient pada RNN biasa
- Cocok untuk data time series dengan pola musiman

### 5.2 Arsitektur Network
```
Input Layer: [batch_size, 12, 2]
     ↓
LSTM Layer: hidden_size=32, num_layers=1
     ↓
Linear Layer: 32 → 1
     ↓
Output: [batch_size, 1]
```

### 5.3 Detail LSTM Cell
Setiap LSTM cell memiliki 3 gate:
- **Forget Gate**: Menentukan informasi mana yang dibuang
- **Input Gate**: Menentukan informasi baru mana yang disimpan
- **Output Gate**: Menentukan output berdasarkan cell state

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)   # Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)   # Input gate
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) # Candidate
C_t = f_t * C_{t-1} + i_t * C̃_t      # Cell state
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)   # Output gate
h_t = o_t * tanh(C_t)                  # Hidden state
```

### 5.4 Hyperparameter
| Parameter | Nilai | Alasan |
|-----------|-------|--------|
| input_size | 2 | SST Indo + Niño 3.4 |
| hidden_size | 32 | Cukup untuk dataset kecil (~150 sampel) |
| num_layers | 1 | Mencegah overfitting |
| lookback | 12 | Menangkap 1 siklus musiman penuh |
| learning_rate | 0.005 | Moderate, dengan decay |
| epochs | 150 | Dengan early stopping via scheduler |

---

## 6. Mekanisme Multivariate Input

### 6.1 Perbedaan Univariate vs Multivariate

**Univariate (modeling.py)**:
```
Input: [SST_t-12, SST_t-11, ..., SST_t-1]
Dimensi: [12, 1]
```

**Multivariate (validation_2013.py)**:
```
Input: [[SST_t-12, Nino_t-12],
        [SST_t-11, Nino_t-11],
        ...
        [SST_t-1,  Nino_t-1]]
Dimensi: [12, 2]
```

### 6.2 Bagaimana LSTM Memproses 2 Feature

Di setiap timestep, LSTM menerima vektor [SST, Niño3.4]. Gate equations menjadi:
```python
x_t = [sst_indonesia, nino34]  # Vektor 2 dimensi

# Forget gate sekarang memiliki bobot untuk kedua feature
f_t = σ(W_f_sst * sst + W_f_nino * nino + W_f_h * h_{t-1} + b_f)
```

Model secara otomatis belajar:
- **W_f_sst**: Seberapa penting SST lokal untuk prediksi
- **W_f_nino**: Seberapa penting sinyal ENSO untuk prediksi

### 6.3 Apa yang Model Pelajari

Selama training, model menemukan pola seperti:
1. "Jika Niño 3.4 meningkat 3 bulan lalu, SST Indonesia cenderung turun sekarang"
2. "Jika SST Indonesia sudah tinggi dan Niño 3.4 netral, SST akan kembali ke normal"
3. "Kombinasi Niño 3.4 tinggi + SST sudah rendah = SST tetap rendah"

### 6.4 Contoh Prediksi

```
Prediksi untuk: Januari 2013
Lookback window:
  Feb 2012: SST=-0.40, Niño=-0.67  (La Niña lemah)
  Mar 2012: SST=-0.30, Niño=-0.61
  ...
  Nov 2012: SST= 0.16, Niño=-0.25
  Dec 2012: SST=-0.25, Niño=-0.25  (Menuju netral)

Model melihat:
  - Niño 3.4 naik dari -0.67 → -0.25 (La Niña melemah)
  - SST Indonesia berfluktuasi

Prediksi: SST sedikit negatif (model belajar ada lag antara Niño dan respons Indonesia)
```

---

## 7. Strategi Validasi

### 7.1 Mengapa Out-of-Sample Penting?

**Split 80/20 Biasa (multivariate_modeling.py)**:
- Data di-shuffle → model bisa "melihat" pola dari masa depan
- Contoh: Training mengandung Dec 2010, test mengandung Jun 2008
- **Tidak realistis** untuk forecasting nyata

**Temporal Split (validation_2013.py)**:
- Training: 2000-2012 (semua data historis)
- Testing: 2013 (data yang belum pernah dilihat)
- **Simulasi nyata**: Bagaimana jika kita melatih model akhir 2012?

### 7.2 Implementasi True Out-of-Sample

```python
# Training data dari CSV (sudah diproses)
train_df = pd.read_csv("sst_indo_clean.csv")  # 2000-2012

# Test data dari NetCDF MENTAH (file berbeda!)
test_data = xr.open_dataset("sst.day.mean.2013.nc")
```

Dengan cara ini, **data 2013 benar-benar tidak pernah dilihat** selama:
- Preprocessing
- Scaling (scaler di-fit hanya pada training data)
- Training model

### 7.3 Handling Lookback untuk Test

Untuk memprediksi Januari 2013, model butuh data 12 bulan sebelumnya:
```
Prediksi Jan 2013: Butuh Feb 2012 - Jan 2013
                          ↑
                   Ini ada di training data!
```

Implementasi:
```python
for i in range(len(test_data)):
    if i < LOOKBACK:
        # Gabungkan akhir training + awal test
        lookback = vstack([train[-LOOKBACK+i:], test[:i]])
    else:
        lookback = test[i-LOOKBACK:i]
```

---

## 8. Interpretasi Hasil

### 8.1 Metrik Evaluasi

| Metrik | Formula | Interpretasi |
|--------|---------|--------------|
| RMSE | √(mean((pred - actual)²)) | Error dalam °C, sensitif terhadap outlier |
| MAE | mean(\|pred - actual\|) | Error rata-rata absolut dalam °C |
| Correlation | corr(pred, actual) | Seberapa baik pola ditangkap (0-1) |

### 8.2 Membaca Visualisasi

**Plot 1: SST Actual vs Predicted**
- Garis biru: Ground truth dari NetCDF 2013
- Garis merah putus-putus: Prediksi model
- Semakin dekat = semakin baik

**Plot 2: Niño 3.4 Index**
- Bar coral (oranye): El Niño (>+0.5°C)
- Bar biru: La Niña (<-0.5°C)
- Bar abu-abu: Netral
- Bandingkan dengan Plot 1 untuk melihat korelasi

**Plot 3: Training Loss**
- Kurva menurun = model belajar
- Titik merah = epoch dengan loss terendah

### 8.3 Validasi Fisik

Pertanyaan untuk validasi:
1. Apakah saat Niño 3.4 positif (El Niño), SST Indonesia negatif? ✓
2. Apakah ada lag yang masuk akal (1-3 bulan)? ✓
3. Apakah model tidak hanya "mengikuti" data kemarin? ✓

---

## Kesimpulan

Proyek ini mendemonstrasikan penerapan Deep Learning untuk prediksi iklim dengan:
1. **Integrasi domain knowledge**: Memanfaatkan telekoneksi ENSO
2. **Multivariate approach**: Menggabungkan local dan remote predictors
3. **Rigorous validation**: True out-of-sample testing
4. **Interpretability**: Visualisasi yang menunjukkan mekanisme prediksi

Model berhasil menangkap pola dasar hubungan ENSO-Indonesia, meskipun untuk aplikasi operasional diperlukan data yang lebih panjang dan model yang lebih kompleks.

---

*Dokumentasi ini dibuat sebagai bagian dari proyek ENSO Forecasting.*
*Terakhir diperbarui: Desember 2024*
