# ==================================================
# UTS BIG DATA TECHNOLOGY
# Smart Campus Attendance Analytics
# ==================================================

# IMPORT LIBRARY
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, sum as _sum, hour
from datetime import datetime, timedelta
import random
import os
import shutil

# ==================================================
# ABSOLUTE PATH CONFIG
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# ==================================================
# INIT SPARK
# ==================================================
spark = SparkSession.builder \
    .appName("UTS_BigData_Attendance") \
    .config("spark.sql.parquet.compression.codec", "snappy") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("🚀 Spark Ready - Memulai Proses...")

# ==================================================
# MEMBERSIHKAN OUTPUT LAMA
# ==================================================
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# GENERATE DUMMY DATA
# ==================================================
buildings = [
    "Fakultas Sains dan Teknologi",
    "Perpustakaan",
    "Auditorium"
]

start_time = datetime(2026, 5, 7, 7, 0)

attendance_data = []

# 100 menit data
for i in range(100):
    for building in buildings:
        attendance_data.append((
            start_time + timedelta(minutes=i),
            building,
            random.randint(20, 300)
        ))

# Membuat Spark DataFrame
attendance_df = spark.createDataFrame(
    attendance_data,
    ["timestamp", "building", "attendance_count"]
)

print("✅ Data berhasil dibuat")

# ==================================================
# SPARK TRANSFORMATION
# ==================================================
# 1. Total mahasiswa per gedung
attendance_total_df = attendance_df.groupBy("building") \
    .agg(_sum("attendance_count").alias("total_attendance"))

# 2. Tren kehadiran per 20 menit
attendance_time_df = attendance_df.groupBy(
    window(col("timestamp"), "20 minutes"),
    col("building")
).agg(
    _sum("attendance_count").alias("total_attendance")
)

# 3. Dataset AI berbasis jam
ml_attendance_df = attendance_df.withColumn(
    "hour",
    hour(col("timestamp"))
)

print("✅ Spark Transformation selesai")

# ==================================================
# SAVE TO PARQUET
# ==================================================
def save_parquet(df, folder_name):
    path = os.path.join(OUTPUT_DIR, folder_name)

    print(f"📂 Menyimpan data ke: {path}")

    df.write.mode("overwrite").parquet(path)

try:
    save_parquet(attendance_total_df, "attendance_total")
    save_parquet(attendance_time_df, "attendance_time")
    save_parquet(ml_attendance_df, "ml_attendance")

    print("\n✅ SEMUA DATA BERHASIL DISIMPAN KE FOLDER OUTPUT")

except Exception as e:
    print(f"\n❌ ERROR SAAT MENULIS DATA: {str(e)}")

# ==================================================
# STOP SPARK
# ==================================================
spark.stop()

print("🛑 Spark Session Closed")