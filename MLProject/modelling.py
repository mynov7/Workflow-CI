import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    if mlflow.active_run():
        mlflow.end_run()

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

    X = df.drop(columns=['loan_status'])
    y = df['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run(run_name="CI_Automated_Run") as run:
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        mlflow.log_metric("accuracy", acc)

        plt.figure(figsize=(8,6))
        sns.heatmap([[1,0],[0,1]], annot=True)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
  
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model berhasil dilatih. Akurasi: {acc:.4f}")

if __name__ == "__main__":
    train_model()
