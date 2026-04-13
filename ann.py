import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# =============================
# Baca CSV (DATA KAMU)
# =============================
df = pd.read_csv("tingkat kriminalitas.csv", sep=";")

# bersihkan nama kolom
df.columns = df.columns.str.strip().str.lower()

# SESUAIKAN KOLOM
df = df[['tahun', 'perkara']]  
df.columns = ['Tahun', 'Kriminalitas']

# FIX angka dengan koma
df["Kriminalitas"] = df["Kriminalitas"].astype(str).str.replace(",", "")
df["Kriminalitas"] = df["Kriminalitas"].astype(float)

print(df.head())

# =============================
# Visualisasi (SUDAH DIGANTI)
# =============================
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Tahun"], y=df["Kriminalitas"], color="blue", label="Data Aktual")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Kriminalitas")
plt.title("Pertumbuhan Kriminalitas")
plt.legend()
plt.show()

# =============================
# Normalisasi (SAMA MODUL)
# =============================
scaler = MinMaxScaler()

df_scaled = scaler.fit_transform(df[["Tahun", "Kriminalitas"]])

X = df_scaled[:, 0].reshape(-1, 1)
Y = df_scaled[:, 1]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# =============================
# Model ANN
# =============================
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train, Y_train,
    epochs=200,
    validation_data=(X_test, Y_test),
    verbose=1
)

# =============================
# Evaluasi
# =============================
loss, mae = model.evaluate(X_test, Y_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# =============================
# Prediksi
# =============================
tahun_prediksi = np.array([[2025], [2030]])

tahun_scaled = scaler.transform(
    np.column_stack((tahun_prediksi, np.zeros(len(tahun_prediksi))))
)[:, 0].reshape(-1, 1)

prediksi_scaled = model.predict(tahun_scaled)

prediksi = scaler.inverse_transform(
    np.column_stack((tahun_scaled[:, 0], prediksi_scaled))
)[:, 1]

# =============================
# Output
# =============================
for tahun, jumlah in zip([2025, 2030], prediksi):
    print(f"Prediksi jumlah kriminalitas tahun {tahun}: {int(jumlah)} kasus")

# =============================
# Visualisasi hasil
# =============================
Y_pred = model.predict(X_test)

plt.figure(figsize=(8,5))
plt.scatter(X_test, Y_test, color='blue', label="Data Aktual")
plt.scatter(X_test, Y_pred, color='red', label="Prediksi ANN")
plt.xlabel("Tahun")
plt.ylabel("Jumlah Kriminalitas")
plt.title("Hasil Prediksi ANN vs Data Aktual")
plt.legend()
plt.show()