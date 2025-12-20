import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "loan_data_preprocessed.csv")

    assert os.path.exists(data_path), "Dataset tidak ditemukan"

    df = pd.read_csv(data_path)
    assert "loan_status" in df.columns, "Kolom loan_status tidak ada"

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="CI_Automated_Run"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        assert acc >= 0.7, f"Akurasi terlalu rendah: {acc}"

        mlflow.log_metric("accuracy", acc)
        mlflow.log_params({
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42
        })

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.savefig("confusion_matrix.png")
        plt.close()

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.sklearn.log_model(model, "model")

        print(f"CI sukses — Akurasi: {acc:.4f}")


if __name__ == "__main__":
    train_model()
