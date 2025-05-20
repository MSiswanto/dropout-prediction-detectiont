import streamlit as st
import pandas as pd
import joblib

# Load model dan data
model = joblib.load("dropout_model.pkl")
data = pd.read_csv("cleaned_data_new.csv")

# Pastikan nama kolom bersih
data.columns = data.columns.str.strip()

# Ambil fitur yang digunakan dalam model
X = data.drop("Target", axis=1)
features = X.columns.tolist()

# UI Streamlit
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")

st.title("🎓 Prediksi Mahasiswa Dropout")
st.markdown("Hasil prediksi: 0 (Dropout), 1 (Enrolled), 2 (Graduate). Masukkan informasi berikut untuk memprediksi kemungkinan dropout:")

# Input form
user_input = {}
with st.form("prediction_form"):
    for col in features:
        if str(data[col].dtype) == 'category':
            user_input[col] = st.selectbox(col, options=data[col].cat.categories.tolist())
        elif "int" in str(data[col].dtype) or "float" in str(data[col].dtype):
            min_val = float(data[col].min())
            max_val = float(data[col].max())
            default = float(data[col].mean())
            user_input[col] = st.slider(col, min_value=min_val, max_value=max_val, value=default)
        else:
            user_input[col] = st.text_input(col)

    submitted = st.form_submit_button("🔍 Prediksi Dropout")

# Prediksi
if submitted:
    try:
        input_df = pd.DataFrame([user_input])
        # Ubah kategori sesuai tipe model training
        for col in input_df.columns:
            if col in data.select_dtypes(include="category").columns:
                input_df[col] = pd.Categorical(input_df[col], categories=data[col].cat.categories)
        prediction = model.predict(input_df)[0]
        st.success(f"📢 Prediksi Status Mahasiswa: **{prediction}**")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {e}")
