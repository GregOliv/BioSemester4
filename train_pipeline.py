# app.py
import streamlit as st
import joblib
import numpy as np

# 1. Load pipeline/model
pipe = joblib.load("model_pipeline.pkl")

st.title("Prediksi Risiko PJK Berbasis Genetik")

# 2. Input User
age         = st.number_input("Usia (tahun)",  20, 100, 45)
sys_bp      = st.number_input("Tekanan Sistolik",  90, 200, 120)
dia_bp      = st.number_input("Tekanan Diastolik", 50, 130, 80)
tot_chol    = st.number_input("Total Kolesterol", 100, 400, 200)
hdl_chol    = st.number_input("HDL Kolesterol", 20, 100, 50)
glucose     = st.number_input("Glukosa Darah", 50, 200, 90)
male        = st.selectbox("Jenis Kelamin", ["Laki-laki","Perempuan"])
smoker      = st.checkbox("Perokok Sekarang")
diabetes    = st.checkbox("Diabetes")
prs         = st.slider("Polygenic Risk Score (PRS)", -3.0, 3.0, 0.0, 0.01)

# 3. Predict button
if st.button("Hitung Risiko"):
    # Encode biner
    male_bin = 1 if male=="Laki-laki" else 0
    smoker_bin = int(smoker)
    diabetes_bin = int(diabetes)
    age_prs = age * prs
    sbp_prs = sys_bp * prs

    features = np.array([[age, sys_bp, dia_bp, tot_chol, hdl_chol,
                          glucose, male_bin, smoker_bin, diabetes_bin,
                          prs, age_prs, sbp_prs]])
    proba = pipe.predict_proba(features)[0,1]
    level = ("Rendah","Sedang","Tinggi")[int(proba>0.5)+int(proba>0.75)]
    st.metric("Probabilitas Risiko PJK", f"{proba*100:.1f}%", level)

# 4. Visualisasi (opsional)
if st.checkbox("Tampilkan Grafik SHAP"):
    import shap
    explainer = shap.Explainer(pipe['classifier'], pipe['preprocessor'].transform)
    shap_values = explainer(pipe['preprocessor'].transform(features))
    st.pyplot(shap.summary_plot(shap_values, features, show=False))
