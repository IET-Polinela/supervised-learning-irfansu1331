import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

# Isi NaN dengan median
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
# 4. Polynomial Regression (Degree 2 & 3)
# =============================

# Buat folder simpanan
os.makedirs("PolynomialRegression", exist_ok=True)

def evaluate_polynomial_model(degree):
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.transform(X_test_scaled)

    model_poly = LinearRegression()
    model_poly.fit(X_train_poly, y_train_clean)
    y_pred = model_poly.predict(X_test_poly)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== Polynomial Regression Degree {degree} ===")
    print("MSE:", mse)
    print("R² :", r2)

    # Visualisasi
    residuals = y_test - y_pred
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 3, 1)
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Degree {degree} - Prediksi vs Aktual")

    plt.subplot(1, 3, 2)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Degree {degree} - Residual Plot")

    plt.subplot(1, 3, 3)
    sns.histplot(residuals, kde=True, bins=30)
    plt.title(f"Degree {degree} - Distribusi Residual")

    plt.tight_layout()
    plt.savefig(f"PolynomialRegression/polynomial_degree_{degree}.png")  # Simpan
    plt.show()

    return mse, r2

# Evaluasi Polynomial Regression degree 2 & 3
mse2, r2_2 = evaluate_polynomial_model(degree=2)
mse3, r2_3 = evaluate_polynomial_model(degree=3)
