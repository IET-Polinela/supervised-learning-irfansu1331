import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

# Load dataset
df = pd.read_csv("train.csv")

# Buat folder untuk menyimpan visualisasi scaling
os.makedirs("FeatureScaling", exist_ok=True)

# Ambil fitur numerik
df_numeric = df.select_dtypes(include=['int64', 'float64'])

# Buang kolom target dan ID
X = df_numeric.drop(columns=["Id", "SalePrice"])

# -------------------------------
# 1. Original (Tanpa Scaling)
# -------------------------------
plt.figure(figsize=(12, 4))
sns.boxplot(data=X)
plt.title("Boxplot - Original (Tanpa Scaling)")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("FeatureScaling/boxplot_original.png")  # Simpan gambar
plt.show()

# -------------------------------
# 2. StandardScaler
# -------------------------------
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)

plt.figure(figsize=(12, 4))
sns.boxplot(data=pd.DataFrame(X_standard, columns=X.columns))
plt.title("Boxplot - StandardScaler")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("FeatureScaling/boxplot_standard_scaler.png")  # Simpan gambar
plt.show()

# -------------------------------
# 3. MinMaxScaler
# -------------------------------
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)

plt.figure(figsize=(12, 4))
sns.boxplot(data=pd.DataFrame(X_minmax, columns=X.columns))
plt.title("Boxplot - MinMaxScaler")
plt.xticks([], [])
plt.tight_layout()
plt.savefig("FeatureScaling/boxplot_minmax_scaler.png")  # Simpan gambar
plt.show()
