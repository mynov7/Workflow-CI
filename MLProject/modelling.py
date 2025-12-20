import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_model():
    df = pd.read_csv("loan_data_preprocessed.csv")
    print("Daftar kolom yang ditemukan:", df.columns.tolist())

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ❗ JANGAN start_run di sini
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Training selesai. Akurasi:", acc)

if __name__ == "__main__":
    train_model()
