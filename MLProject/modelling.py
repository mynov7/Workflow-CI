import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    file_name = 'loan_data_preprocessed.csv'
    if os.path.exists(file_name):
        path = file_name
    else:
        path = os.path.join('MLProject', file_name)
    
    try:
        df = pd.read_csv(path)
        print(f"Berhasil memuat data dari: {path}")
    except Exception as e:
        print(f"Gagal memuat data: {e}")
        return

    # 2. Persiapan Data
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. Pengaturan MLflow
    with mlflow.start_run(nested=True) as run:
        # Model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        
        # Log Parameter dan Metrik
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        
        # Pembuatan Confusion Matrix (Artefak)
        plt.figure(figsize=(8,6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        # Simpan Model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Run ID: {run.info.run_id}")
        print(f"Model berhasil dilatih. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    train_model()
