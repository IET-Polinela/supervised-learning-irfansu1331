import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# =============================
# 1. Load dan Preprocessing Data
# =============================
df = pd.read_csv("train.csv")

# Label Encoding
df_encoded = df.copy()
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df_encoded[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Imputasi NaN dengan median
df_encoded = df_encoded.fillna(df_encoded.median(numeric_only=True))

# Fitur dan target
X = df_encoded.drop(columns=["SalePrice", "Id"])
y = df_encoded["SalePrice"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 2. Outlier Handling
# =============================
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
mask = ~((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(axis=1)
X_train_clean = X_train[mask]
y_train_clean = y_train[mask]

# =============================
# 3. Feature Scaling
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_clean)
X_test_scaled = scaler.transform(X_test)

# =============================
# 4. KNN Regression
# =============================

# Buat folder penyimpanan
os.makedirs("KNNRegression", exist_ok=True)

def evaluate_knn(k):
    model_knn = KNeighborsRegressor(n_neighbors=k)
    model_knn.fit(X_train_scaled, y_train_clean)
    y_pred = model_knn.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== KNN Regression (k={k}) ===")
    print("MSE:", mse)
    print("R² :", r2)

    # Visualisasi
    residuals = y_test - y_pred
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"k={k} - Prediksi vs Aktual")

    plt.subplot(1, 3, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"k={k} - Residual Plot")

    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"k={k} - Distribusi Residual")

    plt.tight_layout()
    plt.savefig(f"KNNRegression/knn_k_{k}.png")  # Simpan gambar
    plt.show()

    return mse, r2

# Evaluasi untuk K = 3, 5, 7
mse_3, r2_3 = evaluate_knn(k=3)
mse_5, r2_5 = evaluate_knn(k=5)
mse_7, r2_7 = evaluate_knn(k=7)
