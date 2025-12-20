import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def train_model():
    df = pd.read_csv("loan_data_preprocessed.csv")

    print("Daftar kolom yang ditemukan:", df.columns.tolist())

    X = df.drop("loan_status", axis=1)
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # SET experiment SAJA (boleh)
    mlflow.set_experiment("Loan_Experiment")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"Model Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_metric("accuracy", acc)

    # Log model untuk CI + Docker
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="Loan_Experiment"
    )

    joblib.dump(model, "model_loan.pkl")
    print("Model berhasil disimpan")

if __name__ == "__main__":
    train_model()
