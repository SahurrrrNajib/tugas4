import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

app = Flask(__name__)

# =============================
# LOAD & TRAIN MODEL
# =============================
df = pd.read_csv("tingkat kriminalitas.csv", sep=";")
df.columns = df.columns.str.strip().str.lower()

df = df[['tahun', 'perkara']]
df.columns = ['Tahun', 'Kriminalitas']

df["Kriminalitas"] = df["Kriminalitas"].astype(str).str.replace(",", "")
df["Kriminalitas"] = df["Kriminalitas"].astype(float)

scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["Tahun", "Kriminalitas"]])

X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, Y_train, epochs=200, verbose=0)

# =============================
# ROUTES
# =============================
@app.route("/", methods=["GET", "POST"])
def index():
    prediksi = None
    
    if request.method == "POST":
        tahun_input = int(request.form["tahun"])

        tahun_scaled = scaler.transform(
            np.column_stack(([tahun_input], [0]))
        )[:, 0].reshape(-1, 1)

        prediksi_scaled = model.predict(tahun_scaled)
        hasil = scaler.inverse_transform(
            np.column_stack((tahun_scaled[:, 0], prediksi_scaled))
        )[:, 1]

        prediksi = int(hasil[0])

        # =============================
        # BUAT GRAFIK
        # =============================
        plt.figure(figsize=(8,5))
        plt.scatter(df["Tahun"], df["Kriminalitas"], label="Data Aktual")
        plt.scatter(tahun_input, prediksi, color="red", label="Prediksi")
        plt.xlabel("Tahun")
        plt.ylabel("Jumlah Kriminalitas")
        plt.legend()
        plt.title("Prediksi Kriminalitas")
        plt.savefig("static/grafik.png")
        plt.close()

    return render_template("index.html", prediksi=prediksi)

if __name__ == "__main__":
    app.run(debug=True)