import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay

# Argument parser for MLflow Project
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Load dataset
data = pd.read_csv(args.data_path)

# Preprocessing (bisa disesuaikan)
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():
    # Parameter logging
    n_estimators = 100
    max_depth = 5
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # Model training
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    preds = model.predict(X_test)

    # Manual metrics logging (lebih dari autolog!)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)

    # Save model to file manually (for GitHub Actions upload)
    os.makedirs("outputs", exist_ok=True)
    model_path = "outputs/stroke_model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model_artifacts")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix")
    cm_path = "outputs/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path, artifact_path="plots")

    # Log the model with MLflow for Docker building
    mlflow.sklearn.log_model(model, artifact_path="model")

print("Training complete. Artifacts saved.")
