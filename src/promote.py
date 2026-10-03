import mlflow

# ==========================================
# CONFIGURATION MLFLOW
# ==========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "LinearRegressionModel"
ALIAS = "champion"
EXPERIMENT_ID = "1"

# ==========================================
# CLIENT MLFLOW
# ==========================================

client = mlflow.MlflowClient()

# ==========================================
# RÉCUPÉRER LES LOGGED MODELS
# ==========================================

models = client.search_logged_models(
    experiment_ids=[EXPERIMENT_ID]
)

print("Nombre de Logged Models :", len(models))

# ==========================================
# RÉCUPÉRER LE DERNIER MODÈLE
# ==========================================

latest_model = models[0]

print()
print("DERNIER LOGGED MODEL")
print("====================")
print("Model ID :", latest_model.model_id)
print("Run ID   :", latest_model.source_run_id)
print("URI      :", latest_model.model_uri)

# ==========================================
# MÉTRIQUES DU NOUVEAU MODÈLE
# ==========================================

latest_run = client.get_run(
    latest_model.source_run_id
)

latest_r2 = latest_run.data.metrics["r2"]
latest_mae = latest_run.data.metrics["mae"]
latest_rmse = latest_run.data.metrics["rmse"]

print()
print("MÉTRIQUES NOUVEAU MODÈLE")
print("=========================")
print("MAE  :", latest_mae)
print("RMSE :", latest_rmse)
print("R²   :", latest_r2)

# ==========================================
# RÉCUPÉRER LE MODÈLE CHAMPION
# ==========================================

champion = client.get_model_version_by_alias(
    MODEL_NAME,
    ALIAS
)

print()
print("MODÈLE CHAMPION")
print("================")
print("Version :", champion.version)
print("Run ID  :", champion.run_id)

# ==========================================
# MÉTRIQUES DU CHAMPION
# ==========================================

champion_run = client.get_run(
    champion.run_id
)

champion_r2 = champion_run.data.metrics["r2"]
champion_mae = champion_run.data.metrics["mae"]
champion_rmse = champion_run.data.metrics["rmse"]

print()
print("MÉTRIQUES CHAMPION")
print("===================")
print("MAE  :", champion_mae)
print("RMSE :", champion_rmse)
print("R²   :", champion_r2)

# ==========================================
# COMPARAISON
# ==========================================

print()
print("COMPARAISON")
print("===========")

print("Nouveau modèle")
print("R²   :", latest_r2)
print("MAE  :", latest_mae)
print("RMSE :", latest_rmse)

print()
print("Champion actuel")
print("R²   :", champion_r2)
print("MAE  :", champion_mae)
print("RMSE :", champion_rmse)

# ==========================================
# DÉCISION
# ==========================================

print()
print("DÉCISION")
print("========")

if latest_r2 > champion_r2:

    print("Nouveau modèle meilleur.")
    print("Promotion vers @champion nécessaire.")

    # ==========================================
    # CRÉER UNE NOUVELLE VERSION
    # ==========================================

    new_version = client.create_model_version(
        name=MODEL_NAME,
        source=latest_model.model_uri
    )

    print()
    print("NOUVELLE VERSION")
    print("=================")
    print("Version :", new_version.version)

    # ==========================================
    # DÉPLACER L'ALIAS @champion
    # ==========================================

    client.set_registered_model_alias(
        MODEL_NAME,
        ALIAS,
        new_version.version
    )

    print()
    print("@champion → Version", new_version.version)

else:

    print("Le champion actuel reste meilleur ou égal.")
    print("Aucune promotion.")