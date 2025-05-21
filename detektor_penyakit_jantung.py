import streamlit as st
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

data_demo = {
    'usia': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57, 50, 60, 58, 62, 65, 54, 48, 51, 45, 53],
    'jenis_kelamin': [1,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,0],
    'tekanan_sistolik': [145,130,130,120,120,140,140,120,172,150,138,135,124,142,148,131,140,129,134,136],
    'tekanan_diastolik': [85,85,83,70,68,84,90,75,89,80,86,78,75,88,90,82,85,80,77,79],
    'kolesterol_total': [233,250,204,236,354,192,294,263,199,168,210,230,220,215,198,240,250,225,218,210],
    'glukosa': [80,80,85,95,100,78,88,92,87,77,90,83,87,89,85,82,90,88,86,84],
    'diabetes': [0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,1,0,0,0],
    'merokok': [0,1,0,0,0,1,0,0,1,0,0,1,0,1,0,0,0,1,0,1],
    'target': [1,1,1,1,1,1,1,0,0,0,0,0,1,0,1,0,1,0,0,1]
}

df = pd.DataFrame(data_demo)
X = df.drop('target', axis=1)
y = df['target']

numerical_features = X.columns.tolist()
preprocessor = ColumnTransformer([('scaler', StandardScaler(), numerical_features)])
model = Pipeline([
    ('prep', preprocessor),
    ('clf', XGBClassifier(eval_metric='logloss', learning_rate=0.1, n_estimators=200))
])

t_X, v_X, t_y, v_y = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(t_X, t_y)

st.set_page_config(page_title="Prediksi Risiko Jantung", layout="centered")
st.title("Prediksi Risiko Penyakit Jantung Koroner")

usia = st.number_input("Usia (tahun)", min_value=1, max_value=120, value=50)
gender_label = st.selectbox("Jenis Kelamin", ["Pria", "Wanita"])
jenis_kelamin = 1 if gender_label == "Pria" else 0

tekanan_sistolik = st.number_input("Tekanan Darah Sistolik (mmHg)", min_value=50, max_value=250, value=120)
tekanan_diastolik = st.number_input("Tekanan Darah Diastolik (mmHg)", min_value=30, max_value=150, value=80)
kolesterol = st.number_input("Kolesterol Total (mg/dL)", min_value=100, max_value=600, value=200)
glukosa = st.number_input("Glukosa (mg/dL)", min_value=50, max_value=500, value=90)

diabetes_label = st.selectbox("Diabetes", ["Tidak", "Ya"])
diabetes = 1 if diabetes_label == "Ya" else 0
merokok_label = st.selectbox("Merokok", ["Tidak", "Ya"])
merokok = 1 if merokok_label == "Ya" else 0

if st.button("Prediksi Risiko"):
    user_data = pd.DataFrame([[usia, jenis_kelamin, tekanan_sistolik, tekanan_diastolik,
                               kolesterol, glukosa, diabetes, merokok]],
                              columns=numerical_features)
    pred = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0][1]

    if pred == 1:
        st.error(f"⚠️ Risiko tinggi ({proba:.2f}) terkena penyakit jantung koroner.")
    else:
        st.success(f"✅ Risiko rendah ({proba:.2f}) terkena penyakit jantung koroner.")

    proba_val = model.predict_proba(v_X)[:,1]
    auc = roc_auc_score(v_y, proba_val)
    prec = precision_score(v_y, model.predict(v_X))
    rec = recall_score(v_y, model.predict(v_X))
    f1 = f1_score(v_y, model.predict(v_X))

    st.markdown("---")
    st.subheader("Evaluasi Model (Data Uji)")
    st.write(f"AUC: {auc:.2f}")
    st.write(f"Precision: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
