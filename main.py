from pathlib import Path
import pandas as pd

from src.utils import preprocess, clean_data, data_clustering, data_prediction
from src.build_model import (
    train_model,
    save_model,
    train_predict_models_by_cluster,
    train_kmeans_from_params,
)
from src.predict import predict
from src.analytics import cluster_summary


# DEFINIÇÃO DE PATHS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "data_raw"
PROCESSED_DATA_DIR = DATA_DIR / "data_processed"

DATA_PATH = RAW_DATA_DIR / "SUPERSTORE_DATASET.csv"

OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "model"

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


# TRATAMENTO DE DADOS INICIAL
df = pd.read_csv(DATA_PATH, sep=",")
df_tratado = clean_data(df)
df_tratado.to_csv(PROCESSED_DATA_DIR / "SUPERSTORE_DATASET_TRATADO.csv", index=False)

print("Dados tratados exportados com sucesso!")

# TRATAMENTO DE DADOS PARA MODELAGEM
df_modelagem = preprocess(df)
df_modelagem.to_csv(PROCESSED_DATA_DIR / "SUPERSTORE_DATASET_MODELAGEM.csv", index=False)

print("Dados para modelagem exportados com sucesso!")


# MODELO DE CLUSTERING (K-MEANS)
X_scaled, df_clustering = data_clustering(df_modelagem)
model_clustering = train_kmeans_from_params(
    X=X_scaled,
    best_kmeans_csv_path=MODEL_DIR / "BEST_KMEANS_PARAMS.csv",
    output_model_path=OUTPUT_DIR / "models_predict" / "kmeans_clustering.joblib",
)
df_clustering["CLUSTER"] = predict(model_clustering, X_scaled)
df_clustering.to_csv(MODEL_DIR / "SUPERSTORE_DATASET_CLUSTERING.csv", index=False)

print("Dados de clustering exportados com sucesso!")


# MODELO DE PREDIÇÃO DE PROFIT POR CLUSTER (REGRESSÃO LINEAR COM REGULARIZAÇÃO)
df_predict = data_prediction(df_clustering)

trained_models = train_predict_models_by_cluster(
    df_predict_by_cluster=df_predict,
    best_models_csv_path=MODEL_DIR / "BEST_MODELS_BY_CLUSTER.csv",
    output_dir=OUTPUT_DIR / "models_predict",
)

print(f"Modelos de predição treinados por cluster: {len(trained_models)}")


print("Pipeline finalizado com sucesso!")
