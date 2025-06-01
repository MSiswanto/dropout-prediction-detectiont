import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Dropout Mahasiswa", layout="wide")

# Load model dan data
model = joblib.load("dropout_model.pkl")
df = pd.read_csv("cleaned_data_new.csv")

# Navigasi
menu = st.sidebar.radio("📁 Navigasi", ["Beranda", "Prediksi", "Visualisasi"])

# ----------------------------
# Halaman Beranda
# ----------------------------
if menu == "Beranda":
    st.title("🎓 Selamat Datang di Aplikasi Prediksi Dropout Mahasiswa")
    st.markdown("""
    Aplikasi ini membantu mendeteksi mahasiswa yang berisiko mengalami **dropout** berdasarkan data akademik dan demografis.
    
    Navigasikan menu di sebelah kiri untuk:
    - Melakukan prediksi risiko dropout
    - Melihat visualisasi data dan fitur penting
    """)
    st.image("https://img.freepik.com/free-vector/flat-design-graduation-ceremony-illustration_23-2149269753.jpg", use_container_width=True)

# ----------------------------
# Halaman Prediksi
# ----------------------------
elif menu == "Prediksi":
    st.title("🔎 Prediksi Risiko Dropout Mahasiswa")

    selected_features = [
        "Age at enrollment",
        "Previous qualification (grade)",
        "Admission grade",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "Curricular units 1st sem (grade)",
        "Curricular units 2nd sem (grade)",
        "Curricular units 1st sem (approved)",
        "Curricular units 2nd sem (approved)"
    ]

    st.sidebar.header("Masukkan Fitur Mahasiswa")
    input_data = {}
    for feature in selected_features:
        if df[feature].dtype in ["int64", "float64"]:
            input_data[feature] = st.sidebar.number_input(f"{feature}", value=float(df[feature].mean()))
        else:
            input_data[feature] = st.sidebar.selectbox(f"{feature}", sorted(df[feature].unique()))

    input_df = pd.DataFrame([input_data])
    if st.sidebar.button("🚀 Prediksi Dropout"):
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.error("❌ Mahasiswa diprediksi berisiko **DROPOUT**.")
        else:
            st.success("✅ Mahasiswa diprediksi **TIDAK dropout**.")

# ----------------------------
# Halaman Visualisasi
# ----------------------------
elif menu == "Visualisasi":
    st.title("📊 Visualisasi Fitur dan Data Mahasiswa")

    st.subheader("🔥 Top 10 Fitur Terpenting Menurut Model")

    selected_features = [
    "Age at enrollment",
    "Previous qualification (grade)",
    "Admission grade",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 1st sem (approved)",
    "Curricular units 2nd sem (approved)"
]

    importances = pd.Series(model.feature_importances_, index=selected_features).sort_values(ascending=False)
    top_features = importances.head(10)

    fig1, ax1 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax1)
    ax1.set_title("Top 10 Fitur yang Mempengaruhi Dropout")
    ax1.set_xlabel("Importance Score")
    st.pyplot(fig1)

    st.subheader("📈 Visualisasi Tambahan")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Distribusi Status Mahasiswa")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='Target', palette='Set2', ax=ax2)
        ax2.set_xlabel("Status")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)

    with col2:
        st.markdown("#### Boxplot Admission Grade by Status")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='Target', y='Admission grade', palette='Pastel1', ax=ax3)
        ax3.set_xlabel("Status")
        ax3.set_ylabel("Admission Grade")
        st.pyplot(fig3)
