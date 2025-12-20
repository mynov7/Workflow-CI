import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    # 1. Load Data
    df = pd.read_csv('loan_data_preprocessed.csv')
    
    X = df.drop('loan_status', axis=1) 
    y = df['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.autolog()
    
    with mlflow.start_run():
        # 3. Training Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
  
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {acc:.2f}")
        print(classification_report(y_test, y_pred))
        
    # 4. Simpan Model 
    joblib.dump(model, 'model_loan.pkl')
    print("Model berhasil disimpan secara lokal sebagai 'model_loan.pkl'")

if __name__ == "__main__":
    train_model()
