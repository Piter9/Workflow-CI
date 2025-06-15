import pandas as pd
import argparse
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Set tracking URI agar menyimpan lokal ke folder mlruns
mlflow.set_tracking_uri("file:./mlruns")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, required=True)
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data_path)

# Fitur dan target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Mulai tracking
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, zero_division=0)

    # Logging manual (bukan autolog)
    mlflow.log_param("model", "RandomForestClassifier")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # Simpan model ke mlruns/0/<run_id>/artifacts/model
    mlflow.sklearn.log_model(model, "model")
