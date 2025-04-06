import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
df = pd.read_csv("train.csv")

# Buat folder untuk menyimpan gambar boxplot
os.makedirs("OutlierHandling", exist_ok=True)

# Ambil hanya kolom numerik
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df_numeric = df[numeric_cols]

# ========== Boxplot Sebelum Outlier ==========
plt.figure(figsize=(20, 10))
sns.boxplot(data=df_numeric, orient="h")
plt.title("Boxplot Sebelum Menghapus Outlier")
plt.tight_layout()
plt.savefig("OutlierHandling/boxplot_sebelum_outlier.png")
plt.show()

# ========== Identifikasi Outlier dengan IQR ==========
Q1 = df_numeric.quantile(0.25)
Q3 = df_numeric.quantile(0.75)
IQR = Q3 - Q1

# Filter: dataset tanpa outlier
df_no_outliers = df_numeric[~((df_numeric < (Q1 - 1.5 * IQR)) | 
                              (df_numeric > (Q3 + 1.5 * IQR))).any(axis=1)]

# ========== Boxplot Setelah Outlier Dihapus ==========
plt.figure(figsize=(20, 10))
sns.boxplot(data=df_no_outliers, orient="h")
plt.title("Boxplot Setelah Menghapus Outlier")
plt.tight_layout()
plt.savefig("OutlierHandling/boxplot_setelah_outlier.png")
plt.show()

# ========== Informasi ==========
print("Jumlah data awal:", df_numeric.shape[0])
print("Jumlah data setelah menghapus outlier:", df_no_outliers.shape[0])
print("Jumlah data yang dihapus:", df_numeric.shape[0] - df_no_outliers.shape[0])
