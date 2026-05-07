# UTS BIG DATA TECHNOLOGY – Smart Campus Attendance Analytics

**Project:** UTS-TBG-230104040059

Program ini dibuat untuk memenuhi tugas **Ujian Tengah Semester (UTS) Mata Kuliah Teknologi Big Data** dengan tema implementasi **Big Data Pipeline menggunakan PySpark, Parquet, Machine Learning, dan Streamlit Dashboard**.

Sistem ini berfokus pada analisis kepadatan mahasiswa kampus menggunakan data simulasi kehadiran mahasiswa pada beberapa gedung untuk mengetahui tren kehadiran serta memprediksi jam sibuk kampus.

---

# 👩‍🎓 Identitas Mahasiswa

| Keterangan | Informasi |
|---|---|
| Mata Kuliah | Teknologi Big Data |
| Dosen Pengampu | Muhayat, S.Ag., M.IT |
| Tema Project | Smart Campus Attendance Analytics |
| Nama Mahasiswa | Siti Magfiratun Warahmah |
| NIM | 230104040059 |
| Kelas | TI23B |
| Semester | Genap 2025/2026 |

---

# 🚀 Deskripsi Program

Program ini merupakan implementasi sistem **Smart Campus Attendance Analytics** berbasis Big Data menggunakan Apache Spark dan Streamlit Dashboard.

Sistem bekerja dengan cara:

- Membuat data simulasi kehadiran mahasiswa
- Memproses data menggunakan Apache Spark (PySpark)
- Melakukan agregasi data berdasarkan gedung dan waktu
- Menyimpan hasil pengolahan ke format Parquet
- Menampilkan dashboard interaktif menggunakan Streamlit dan Plotly
- Melakukan prediksi jumlah mahasiswa menggunakan Linear Regression

Sistem ini digunakan untuk membantu analisis kepadatan mahasiswa kampus dan mengetahui jam tersibuk kampus berdasarkan data kehadiran mahasiswa.

---

# 🏗️ Arsitektur Pipeline

```text
Attendance Data
      ↓
Spark Analytics
      ↓
Parquet Storage
      ↓
AI Prediction
      ↓
Streamlit Dashboard
```

Penjelasan alur pipeline:

1. Data kehadiran mahasiswa dibuat secara simulasi menggunakan Python.
2. Apache Spark digunakan untuk melakukan transformasi dan agregasi data.
3. Data hasil transformasi disimpan dalam format Parquet.
4. Machine Learning digunakan untuk melakukan prediksi jumlah mahasiswa.
5. Dashboard Streamlit digunakan untuk menampilkan visualisasi data secara interaktif.

---

# 🛠️ Teknologi yang Digunakan

- Python 3
- Apache Spark (PySpark)
- Pandas
- Streamlit
- Plotly
- Scikit-Learn
- Linear Regression
- WSL (Linux)
- Visual Studio Code

---

# 📂 Struktur Folder Project

```text
uts-tbg-230104040059
│
├── main_uts_230104040059.py
├── dashboard_230104040059.py
├── output/
│   ├── attendance_total/
│   ├── attendance_time/
│   └── ml_attendance/
│
├── .gitignore
└── README.md
```

---

# ⚙️ Cara Menjalankan Program

## 1️⃣ Aktifkan Virtual Environment

```bash
source venv/bin/activate
```

---

## 2️⃣ Jalankan Spark Engine

```bash
python3 main_uts_230104040059.py
```

Output:
- Data Parquet akan tersimpan di folder `output/`

---

## 3️⃣ Jalankan Dashboard Streamlit

```bash
streamlit run dashboard_230104040059.py
```

---

# 🌐 Akses Dashboard

Buka browser dan akses:

```text
http://localhost:8501
```

---

# 📊 Fitur Dashboard

Dashboard menampilkan:

- Sidebar filter gedung
- KPI total mahasiswa
- Grafik tren kehadiran mahasiswa
- Prediksi jumlah mahasiswa menggunakan AI
- Visualisasi interaktif menggunakan Plotly

Gedung yang dianalisis:

- Fakultas Sains dan Teknologi
- Perpustakaan
- Auditorium

---

# 📦 Output Sistem

Sistem menghasilkan:

- Total mahasiswa seluruh gedung
- Total mahasiswa per gedung
- Grafik tren kehadiran mahasiswa
- Dataset Machine Learning berbasis jam
- Prediksi jumlah mahasiswa berdasarkan jam

Contoh hasil dashboard:

- Total mahasiswa seluruh gedung: **47.495**
- Total mahasiswa Perpustakaan: **15.687**
- Prediksi mahasiswa pukul 12.00: **119 mahasiswa**

---

# 📸 Output Wajib

Dokumentasi yang dihasilkan:

- Script Python
  - `main_uts_230104040059.py`
  - `dashboard_230104040059.py`

- Screenshot:
  - Dashboard Streamlit
  - File Parquet berhasil dibuat
  - Terminal Spark berhasil dijalankan

- Analisis jam sibuk kampus

---

# 📈 Analisis Jam Sibuk Kampus

Berdasarkan hasil visualisasi dashboard, jam tersibuk kampus terjadi pada sekitar pukul **07.40**. Hal tersebut terlihat dari peningkatan jumlah mahasiswa pada grafik tren kehadiran, terutama pada gedung Auditorium dan Fakultas Sains dan Teknologi. Gedung Perpustakaan juga menunjukkan tingkat aktivitas mahasiswa yang cukup tinggi dengan total mahasiswa mencapai **15.687** mahasiswa.

Hasil prediksi menggunakan Linear Regression menunjukkan bahwa jumlah mahasiswa pada pukul **12.00** mengalami penurunan menjadi sekitar **119 mahasiswa**, sehingga aktivitas kampus paling padat terjadi pada pagi hari.

---

# 🎯 Kesimpulan

Implementasi Big Data Pipeline menggunakan PySpark, Parquet, Machine Learning, dan Streamlit Dashboard berhasil dilakukan dengan baik. Sistem mampu melakukan pengolahan data, penyimpanan data, visualisasi interaktif, serta prediksi jumlah mahasiswa untuk membantu analisis kepadatan mahasiswa kampus secara lebih cepat dan informatif.

---

# 📦 Repository GitHub

```text
https://github.com/Siti-Magfiratun-Warahmah/Big-Data-UTS-Siti-Magfiratun-Warahmah-230104040059
```