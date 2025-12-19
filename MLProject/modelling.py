import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    df = pd.read_csv('loan_data_preprocessed.csv')
    
    print("Daftar kolom yang ditemukan:", df.columns.tolist())
    
    X = df.drop('loan_status', axis=1) 
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Mulai Eksperimen dengan MLflow
    mlflow.set_experiment("Loan_Experiment")
    
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prediksi dan Evaluasi
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred))
        
        # Log parameter dan metrik ke MLflow Dashboard
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_metric("accuracy", acc)
        
        # 4. Simpan Model sebagai Artifak di MLflow
        mlflow.sklearn.log_model(model, name="model_loan_artifact")
        
        # 5. Simpan Model secara lokal untuk folder Membangun_model
        joblib.dump(model, 'model_loan.pkl')
        print("Model berhasil disimpan sebagai 'model_loan.pkl'")

if __name__ == "__main__":
    train_model()