import mlflow.sklearn
import pandas as pd

# Connexion à MLflow
mlflow.set_tracking_uri(
    "sqlite:///C:/Users/AINA KEVIN/Desktop/Ml Flow/mlflow.db"
)

# Charger le modèle champion
model = mlflow.sklearn.load_model(
    "models:/LinearRegressionModel@champion"
)

# Nouvelle donnée
X_new = pd.DataFrame({
    "x": [50]
})

# Prédiction
prediction = model.predict(X_new)

print("================================")
print("PRÉDICTION")
print("================================")
print("x =", X_new["x"].iloc[0])
print("y prédit =", prediction[0])