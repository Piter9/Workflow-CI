import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Membangun Model")

# Aktifkan autolog
mlflow.sklearn.autolog()

# Load data
data = pd.read_csv("healthcare-dataset-stroke_preprocessing.csv")

# Bagi fitur dan target
X = data.drop("stroke", axis=1)
y = data["stroke"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tracking eksperimen
with mlflow.start_run():
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    # mlflow.sklearn.log_model(
    #     sk_model=model,
    #     name="model",
    #     input_example=X_test.iloc[:5]
    # )    
    print("Akurasi:", acc)
