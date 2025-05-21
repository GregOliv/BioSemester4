import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Membuat data demo menyerupai data framingham
data_demo = {
    'usia': [63, 37, 41, 56, 57, 57, 56, 44, 52, 57, 50, 60, 58, 62, 65, 54, 48, 51, 45, 53],
    'jenis_kelamin': [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0],  # 1=Pria, 0=Wanita
    'tekanan_sistolik': [145, 130, 130, 120, 120, 140, 140, 120, 172, 150, 138, 135, 124, 142, 148, 131, 140, 129, 134, 136],
    'tekanan_diastolik': [85, 85, 83, 70, 68, 84, 90, 75, 89, 80, 86, 78, 75, 88, 90, 82, 85, 80, 77, 79],
    'kolesterol_total': [233, 250, 204, 236, 354, 192, 294, 263, 199, 168, 210, 230, 220, 215, 198, 240, 250, 225, 218, 210],
    'glukosa': [80, 80, 85, 95, 100, 78, 88, 92, 87, 77, 90, 83, 87, 89, 85, 82, 90, 88, 86, 84],
    'diabetes': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0],  # 0=tidak, 1=ya
    'merokok': [0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1],  # 0=tidak, 1=ya
    'target': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1]
}

# Membuat DataFrame
df = pd.DataFrame(data_demo)

X = df.drop('target', axis=1)
y = df['target']

# Mendefinisikan fitur numerik
fitur_numerik = X.columns.tolist()

# Pipeline preprocessing + model XGBoost
preprocessor = ColumnTransformer([
    ('standarisasi', StandardScaler(), fitur_numerik)
])

model = Pipeline([
    ('preprocessing', preprocessor),
    ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', learning_rate=0.1, n_estimators=200))
])

# Membagi data latih dan data uji
X_latih, X_uji, y_latih, y_uji = train_test_split(X, y, test_size=0.2, random_state=42)

# Melatih model
model.fit(X_latih, y_latih)

# Evaluasi
y_pred = model.predict(X_uji)
auc = roc_auc_score(y_uji, y_pred)
precision = precision_score(y_uji, y_pred)
recall = recall_score(y_uji, y_pred)
f1 = f1_score(y_uji, y_pred)

# Setup Flask app
app = Flask(__name__)

# Template HTML dengan bahasa Indonesia
template_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Prediksi Risiko Penyakit Jantung Koroner</title>
    <style>
        body { font-family: Arial, sans-serif; background: #f7f9fc; margin: 20px; }
        h1 { color: #b71c1c; text-align: center; }
        form { background: white; max-width: 500px; margin: auto; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        label { display: block; margin-top: 10px; font-weight: bold; }
        input[type=number], select { width: 100%; padding: 8px; margin-top: 5px; border: 1px solid #ccc; border-radius: 4px; }
        input[type=submit] { margin-top: 20px; background: #b71c1c; color: white; border: none; padding: 10px; width: 100%; font-size: 16px; cursor: pointer; border-radius: 4px; }
        input[type=submit]:hover { background: #d32f2f; }
        .hasil { max-width: 500px; margin: auto; margin-top: 20px; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; font-size: 1.2em; }
        .risiko { color: red; }
        .aman { color: green; }
        .evaluasi { max-width: 500px; margin: auto; padding: 15px; font-size: 0.9em; text-align: center; color: #444; }
    </style>
</head>
<body>
    <h1>Prediksi Risiko Penyakit Jantung Koroner</h1>
    <form method="POST">
        <label>Usia (tahun):</label>
        <input type="number" name="usia" min="1" max="120" required>

        <label>Jenis Kelamin:</label>
        <select name="jenis_kelamin" required>
            <option value="1">Pria</option>
            <option value="0">Wanita</option>
        </select>

        <label>Tekanan Darah Sistolik (mmHg):</label>
        <input type="number" name="tekanan_sistolik" min="50" max="250" required>

        <label>Tekanan Darah Diastolik (mmHg):</label>
        <input type="number" name="tekanan_diastolik" min="30" max="150" required>

        <label>Kolesterol Total (mg/dL):</label>
        <input type="number" name="kolesterol_total" min="100" max="600" required>

        <label>Glukosa (mg/dL):</label>
        <input type="number" name="glukosa" min="50" max="500" required>

        <label>Diabetes (0=tidak, 1=ya):</label>
        <select name="diabetes" required>
            <option value="0">Tidak</option>
            <option value="1">Ya</option>
        </select>

        <label>Merokok (0=tidak, 1=ya):</label>
        <select name="merokok" required>
            <option value="0">Tidak</option>
            <option value="1">Ya</option>
        </select>

        <input type="submit" value="Prediksi Risiko">
    </form>

    {% if hasil is not none %}
        <div class="hasil {% if hasil == 1 %}risiko{% else %}aman{% endif %}">
            {% if hasil == 1 %}
                ⚠️ Risiko tinggi terkena penyakit jantung koroner. Silakan konsultasikan dengan dokter.
            {% else %}
                ✅ Risiko rendah terkena penyakit jantung koroner. Jaga pola hidup sehat.
            {% endif %}
        </div>

        <div class="evaluasi">
            <strong>Evaluasi Model (data uji):</strong><br>
            AUC: {{ auc }}<br>
            Precision: {{ precision }}<br>
            Recall: {{ recall }}<br>
            F1-Score: {{ f1 }}<br>
        </div>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def prediksi():
    hasil = None
    if request.method == 'POST':
        try:
            fitur = ['usia', 'jenis_kelamin', 'tekanan_sistolik', 'tekanan_diastolik', 'kolesterol_total', 'glukosa', 'diabetes', 'merokok']
            input_user = []
            for f in fitur:
                val = request.form.get(f)
                if val is None:
                    return f"Error: Field '{f}' tidak ditemukan."
                input_user.append(float(val))
            data_input = pd.DataFrame([input_user], columns=fitur)
            prediksi_model = model.predict(data_input)[0]
            hasil = int(prediksi_model)
        except Exception as e:
            return f"Terjadi kesalahan saat prediksi: {str(e)}"
    return render_template_string(template_html,
                                  hasil=hasil,
                                  auc=f"{auc:.2f}",
                                  precision=f"{precision:.2f}",
                                  recall=f"{recall:.2f}",
                                  f1=f"{f1:.2f}")

if __name__ == '__main__':
    app.run(debug=True)