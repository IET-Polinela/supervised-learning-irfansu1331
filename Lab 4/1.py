import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv('train.csv')

# Tampilkan 5 baris pertama
print("=== 5 Baris Pertama Dataset ===")
print(df.head())

# Ambil kolom numerik
df_num = df.select_dtypes(include=[np.number])

# Statistik deskriptif
print("\n=== Statistik Deskriptif Lengkap (Kolom Numerik Saja) ===")
desc = df_num.describe().T
desc['median'] = df_num.median()
desc['Q1'] = df_num.quantile(0.25)
desc['Q2'] = df_num.quantile(0.50)
desc['Q3'] = df_num.quantile(0.75)
desc['non_null_count'] = df_num.count()
desc['missing'] = df_num.isnull().sum()

# Tampilkan statistik yang diminta
print(desc[['mean', 'median', 'std', 'min', 'Q1', 'Q2', 'Q3', 'max', 'non_null_count', 'missing']])

# Tampilkan kolom dengan nilai hilang dari seluruh dataset
print("\n=== Kolom dengan Nilai Hilang (Semua Tipe Data) ===")
missing_cols = df.columns[df.isnull().any()]
print(df[missing_cols].isnull().sum())
