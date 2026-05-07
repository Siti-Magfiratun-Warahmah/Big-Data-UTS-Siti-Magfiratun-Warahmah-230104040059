# ==================================================
# DASHBOARD STREAMLIT
# Smart Campus Attendance Analytics
# ==================================================

# IMPORT LIBRARY
import streamlit as st
from pyspark.sql import SparkSession
import plotly.express as px
import pandas as pd
from sklearn.linear_model import LinearRegression
import os

# ==================================================
# CONFIG
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

st.set_page_config(
    page_title="Smart Campus Attendance Analytics",
    layout="wide"
)

st.title("🎓 Smart Campus Attendance Analytics Dashboard")

# ==================================================
# INIT SPARK
# ==================================================
@st.cache_resource
def get_spark():
    return SparkSession.builder \
        .appName("Dashboard_App") \
        .getOrCreate()

spark = get_spark()

# ==================================================
# LOAD PARQUET
# ==================================================
def load_parquet(folder_name):

    path = os.path.join(OUTPUT_DIR, folder_name)

    if not os.path.exists(path):
        st.error(f"❌ Folder data '{folder_name}' tidak ditemukan!")
        st.stop()

    return spark.read.parquet(path).toPandas()

try:
    total_df = load_parquet("attendance_total")
    time_df = load_parquet("attendance_time")
    ml_df = load_parquet("ml_attendance")

except Exception as e:
    st.error(f"❌ Gagal memuat data: {e}")
    st.stop()

# ==================================================
# SIDEBAR FILTER
# ==================================================
st.sidebar.title("📌 Filter Dashboard")

buildings = total_df["building"].unique()

selected_building = st.sidebar.selectbox(
    "Pilih Gedung",
    buildings
)

filtered_total = total_df[
    total_df["building"] == selected_building
]

# ==================================================
# KPI METRICS
# ==================================================
st.subheader("📊 Key Performance Indicators")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Total Mahasiswa (Semua Gedung)",
        int(total_df["total_attendance"].sum())
    )

with col2:
    st.metric(
        f"Total Mahasiswa di {selected_building}",
        int(filtered_total["total_attendance"].sum())
    )

# ==================================================
# VISUALIZATION
# ==================================================
st.markdown("---")

st.subheader("📈 Grafik Tren Kehadiran")

# Memperbaiki format waktu
time_df["start_time"] = time_df["window"].apply(
    lambda x: x["start"] if isinstance(x, dict) else x.start
)

fig = px.line(
    time_df,
    x="start_time",
    y="total_attendance",
    color="building",
    title="Tren Kehadiran Mahasiswa"
)

st.plotly_chart(fig, use_container_width=True)

# ==================================================
# MACHINE LEARNING
# ==================================================
st.markdown("---")

st.subheader("🤖 AI Prediction (Linear Regression)")

# Data AI
X = ml_df[["hour"]]
y = ml_df["attendance_count"]

# Training model
model = LinearRegression()
model.fit(X, y)

# Slider prediksi
hour_input = st.slider(
    "Pilih Jam Prediksi",
    0,
    23,
    12
)

prediction = model.predict([[hour_input]])

st.success(
    f"📌 Prediksi jumlah mahasiswa pada jam "
    f"{hour_input}:00 adalah sekitar "
    f"{int(prediction[0])} mahasiswa"
)