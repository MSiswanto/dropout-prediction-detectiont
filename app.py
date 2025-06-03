import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Konfigurasi halaman
st.set_page_config(page_title="Aplikasi Dropout Mahasiswa", layout="wide")

# Tambahan CSS untuk sidebar
st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-size: 24px;
        font-weight: bold;
        color: #4B8BBE;
        margin-bottom: 10px;
    }
    .sidebar-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ccc;
    }
    </style>
""", unsafe_allow_html=True)

# Tampilkan judul sidebar
st.sidebar.markdown('<div class="sidebar-title">📁 Navigasi</div>', unsafe_allow_html=True)
with st.sidebar:
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    menu = st.radio("", ["🏠 Beranda", "🔍 Prediksi", "📊 Visualisasi"])
    st.markdown('</div>', unsafe_allow_html=True)

# Load model dan data
model = joblib.load("dropout_model.pkl")
df = pd.read_csv("cleaned_data_new.csv")

# Navigasi
#menu = st.sidebar.radio("📁 Navigasi", ["Beranda", "Prediksi", "Visualisasi"])

# ----------------------------
# Halaman Beranda
# ----------------------------
if menu == "🏠 Beranda":
    st.title("🎓 Aplikasi Prediksi Dropout Mahasiswa")
    st.markdown("""
    Aplikasi ini membantu mendeteksi mahasiswa yang berisiko mengalami **dropout** berdasarkan data akademik dan demografis.

    Navigasikan menu di sebelah kiri untuk:
    - Mengamati fitur penting yang berpengaruh pada dropout 
    - Melakukan prediksi risiko dropout
    - Melihat visualisasi data
    """)
    
    st.subheader("🔥 Top 10 Fitur Terpenting Berpengaruh Pada Dropout")
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

    fig0, ax0 = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis', ax=ax0)
    ax0.set_title("Top 10 Fitur yang Mempengaruhi Dropout")
    ax0.set_xlabel("Importance Score")
    ax0.set_ylabel("Fitur")
    st.pyplot(fig0)

    #st.image("https://img.freepik.com/free-vector/flat-design-graduation-ceremony-illustration_23-2149269753.jpg", use_container_width=True)


# ----------------------------
# Halaman Prediksi
# ----------------------------
elif menu == "🔍 Prediksi":
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

    st.subheader("🧾 Masukkan Fitur Mahasiswa")

    # Layout input menjadi 2 kolom
    cols = st.columns(2)
    input_data = {}
    for i, feature in enumerate(selected_features):
        with cols[i % 2]:
            if df[feature].dtype in ["int64", "float64"]:
                input_data[feature] = st.number_input(f"**{feature}**", value=float(df[feature].mean()), key=feature)
            else:
                input_data[feature] = st.selectbox(f"**{feature}**", sorted(df[feature].unique()), key=feature)

    input_df = pd.DataFrame([input_data])

    # Threshold untuk visualisasi (nilai ini bisa kamu sesuaikan)
    thresholds = {
        "Age at enrollment": 20,
        "Previous qualification (grade)": 12,
        "Admission grade": 140,
        "Curricular units 1st sem (grade)": 10,
        "Curricular units 2nd sem (grade)": 10,
    }

    # Tombol prediksi
    if st.button("🚀 Prediksi Dropout"):
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.error("❌ Mahasiswa diprediksi berisiko **DROPOUT**.")
        else:
            st.success("✅ Mahasiswa diprediksi **TIDAK dropout**.")

        # Visualisasi Input vs Threshold
        st.subheader("📉 Perbandingan Nilai Input vs Batas Dropout")

        #import matplotlib.pyplot as plt
        #import seaborn as sns

        input_vals = {k: v for k, v in input_data.items() if k in thresholds}
        features = list(input_vals.keys())
        values = list(input_vals.values())
        threshold_vals = [thresholds[f] for f in features]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=features, y=values, palette='Blues_d', ax=ax)
        ax.plot(features, threshold_vals, color='red', marker='o', linestyle='--', label='Threshold')

        # Tambahkan anotasi nilai
        for i, val in enumerate(values):
            ax.text(i, val + 0.5, f"{val:.1f}", ha='center', va='bottom')

        ax.set_ylabel("Nilai Fitur")
        ax.set_title("Nilai Input Mahasiswa vs Threshold Dropout")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right') 
        ax.legend()
        st.pyplot(fig)

# ----------------------------
# Halaman Visualisasi
# ----------------------------
elif menu == "📊 Visualisasi":

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
    ax1.set_ylabel("Fitur")
    st.pyplot(fig1)

    st.subheader("📈 Visualisasi Tambahan")
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("#### Distribusi Status Mahasiswa")
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='Target', palette='Set2', ax=ax2)
        ax2.set_xlabel("Status")
        ax2.set_ylabel("Jumlah")
        st.pyplot(fig2)

    with col5:
        st.markdown("#### Boxplot Admission Grade by Status")
        fig3, ax3 = plt.subplots()
        sns.boxplot(data=df, x='Target', y='Admission grade', palette='Pastel1', ax=ax3)
        ax3.set_xlabel("Status")
        ax3.set_ylabel("Admission Grade")
        st.pyplot(fig3)

    with col6:
        st.markdown("#### Distribusi Berdasarkan Umur")
        fig4, ax4 = plt.subplots()
        sns.barplot(data=df, x='Target', y='Age at enrollment', palette='Set2', ax=ax4)
        ax4.set_xlabel("Target")
        ax4.set_ylabel("Age")
        st.pyplot(fig4)


    st.subheader("📈 Visualisasi Tambahan-1")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Distribusi by Gender")
        fig5, ax5 = plt.subplots()
        #sns.barplot(data=df, x='Gender', y='Target', palette='Set2', ax=ax5)
        sns.countplot(data=df, x='Gender', palette='Set2', ax=ax5)
        ax5.set_xlabel("Gender")
        ax5.set_ylabel("Status")
        st.pyplot(fig5)

    with col2:
        st.markdown("#### Curricular 1st Sem")
        fig6, ax6 = plt.subplots()
        sns.boxplot(data=df, x='Target', y='Curricular units 1st sem (grade)', palette='Pastel1', ax=ax6)
        ax6.set_xlabel("Status")
        ax6.set_ylabel("Curricular 1st Sem")
        st.pyplot(fig6)

    with col3:
        st.markdown("#### Previous Qualification")
        fig7, ax7 = plt.subplots()
        #sns.countplot(data=df, x='Target', palette='Set2', ax=ax7)
        sns.barplot(data=df, x='Target', y='Previous qualification (grade)', palette='Set2', ax=ax7)
        ax7.set_xlabel("Target")
        ax7.set_ylabel("Previous Qualification Grade")
        st.pyplot(fig7)
