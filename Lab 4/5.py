import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================
# 1. Load dan Preprocessing Data
# =============================
df = pd.read_csv("train.csv")

# Encoding fitur kategorikal
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Isi missing value dengan median
df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))

# Pisahkan fitur dan target
X = df_encoded.drop(columns=["SalePrice", "Id"])
y = df_encoded["SalePrice"]

# Split data (dengan outlier)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 2. Model dengan Outlier
# =============================
model_outlier = LinearRegression()
model_outlier.fit(X_train, y_train)

y_pred_outlier = model_outlier.predict(X_test)
mse_outlier = mean_squared_error(y_test, y_pred_outlier)
r2_outlier = r2_score(y_test, y_pred_outlier)

print("=== Model dengan Outlier ===")
print("MSE:", mse_outlier)
print("R² :", r2_outlier)

# =============================
# 3. Model tanpa Outlier + Scaling
# =============================
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(axis=1)

X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

# Model
model_clean = LinearRegression()
model_clean.fit(X_train_scaled, y_train_clean)

y_pred_clean = model_clean.predict(X_test_scaled)
mse_clean = mean_squared_error(y_test, y_pred_clean)
r2_clean = r2_score(y_test, y_pred_clean)

print("\n=== Model tanpa Outlier + Scaling ===")
print("MSE:", mse_clean)
print("R² :", r2_clean)

# =============================
# 4. Visualisasi Hasil Prediksi
# =============================
# Buat folder untuk menyimpan hasil
os.makedirs("LinearRegression", exist_ok=True)

def plot_evaluation(y_true, y_pred, title_prefix, filename_prefix):
    residuals = y_true - y_pred

    plt.figure(figsize=(16, 4))

    # Scatter plot: Prediksi vs Aktual
    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"{title_prefix} - Actual vs Predicted")

    # Residual Plot
    plt.subplot(1, 3, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"{title_prefix} - Residual Plot")

    # Distribusi Residual
    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"{title_prefix} - Residual Distribution")

    plt.tight_layout()
    plt.savefig(f"LinearRegression/{filename_prefix}.png")  # Simpan gambar
    plt.show()

# Simpan visualisasi model dengan outlier
plot_evaluation(y_test, y_pred_outlier, "Dengan Outlier", "model_dengan_outlier")

# Simpan visualisasi model tanpa outlier + scaling
plot_evaluation(y_test, y_pred_clean, "Tanpa Outlier + Scaling", "model_tanpa_outlier_scaling")
