import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. CHARGEMENT DES DONNÉES
# ============================================================

df = pd.read_excel("data/raw/synthetic_data.xlsx")

print("Dataset :")
print(df.head())

print("\nShape :", df.shape)


# ============================================================
# 2. VARIABLES
# ============================================================

X = df[["x"]]
y = df["y"]


# ============================================================
# 3. TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain :", X_train.shape)
print("Test  :", X_test.shape)


# ============================================================
# 4. CONFIGURATION MLFLOW
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Linear Regression")


# ============================================================
# 5. ENTRAÎNEMENT + TRACKING MLFLOW
# ============================================================

with mlflow.start_run():

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


    # --------------------------------------------------------
    # Paramètres
    # --------------------------------------------------------

    mlflow.log_param("model", "LinearRegression")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)


    # --------------------------------------------------------
    # Métriques
    # --------------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(y_test, y_pred)


    mlflow.log_metric("mae", mae)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)


    # --------------------------------------------------------
    # Modèle
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="model"
    )


    # ========================================================
    # RÉSULTATS
    # ========================================================

    print("\n==============================")
    print("RÉSULTATS")
    print("==============================")

    print("Coefficient :", model.coef_[0])
    print("Intercept   :", model.intercept_)
    print("MAE         :", mae)
    print("RMSE        :", rmse)
    print("R²          :", r2)

    print("\nRun MLflow :", mlflow.active_run().info.run_id)