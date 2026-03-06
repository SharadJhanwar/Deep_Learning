import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

# Suppress Warnings
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
if not os.path.exists("customer_churn.csv"):
    print("Error: customer_churn.csv not found")
    exit(1)

df = pd.read_csv("customer_churn.csv")
df.drop('customerID',axis='columns',inplace=True)

# Remove space rows
df1 = df[df.TotalCharges!=' '].copy() # Use copy to avoid SettingWithCopyWarning for validation
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)

# Fixes applied (simulating the fixed notebook)
yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
                  'DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling','Churn']
for col in yes_no_columns:
    df1[col] = df1[col].replace({'Yes': 1,'No': 0})

df1['gender'] = df1['gender'].replace({'Female':1,'Male':0})

# Fix get_dummies
df2 = pd.get_dummies(data=df1, columns=['InternetService','Contract','PaymentMethod'], dtype=int)

cols_to_scale = ['tenure','MonthlyCharges','TotalCharges']
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

X = df2.drop('Churn',axis='columns')
y = df2['Churn']

print("X_train numeric check:")
non_numeric = X.select_dtypes(exclude=[np.number])
if not non_numeric.empty:
    print(f"FAILED: Found non-numeric columns: {non_numeric.columns.tolist()}")
    exit(1)
else:
    print("SUCCESS: All columns are numeric.")

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

# Try model
model = keras.Sequential([
    keras.layers.Dense(26, input_shape=(26,), activation='relu'),
    keras.layers.Dense(15, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("Attempting training...")
try:
    model.fit(X_train, y_train, epochs=1, verbose=1)
    print("Training successful!")
except Exception as e:
    print(f"Training failed: {e}")
    exit(1)
