import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('train.csv')

# Encoding fitur kategorikal (non-numerik) menggunakan One-Hot Encoding
df_encoded = pd.get_dummies(df)

# Pisahkan fitur dan target
X = df_encoded.drop(columns=["SalePrice", "Id"])  # 'Id' juga tidak relevan
Y = df_encoded["SalePrice"]

# Split data: 80% training, 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Tampilkan dimensi hasil split
print("Ukuran X_train:", X_train.shape)
print("Ukuran X_test :", X_test.shape)
print("Ukuran Y_train:", Y_train.shape)
print("Ukuran Y_test :", Y_test.shape)
