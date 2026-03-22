import ast
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNet, Lasso, Ridge
import joblib


def train_model(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=0)
    model.fit(X)
    return model

def save_model(model, path="model/model.joblib"):
    joblib.dump(model, path)


def train_kmeans_from_params(X, best_kmeans_csv_path, output_model_path):
    """
    Treina KMeans a partir de um CSV com colunas: MODEL, BEST_PARAMS.
    """
    kmeans_df = pd.read_csv(best_kmeans_csv_path)

    if kmeans_df.empty:
        raise ValueError("Arquivo BEST_KMEANS_PARAMS.csv está vazio")

    row = kmeans_df.iloc[0]
    model_name = str(row.get("MODEL", "KMEANS")).strip().upper()
    params = _parse_params(row.get("BEST_PARAMS", {}))

    if model_name != "KMEANS":
        raise ValueError(f"Modelo não suportado para clustering: {model_name}")

    model = KMeans(**params)
    model.fit(X)

    output_path = Path(output_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_model(model, output_path)

    return model


def _parse_params(params_value):
    if isinstance(params_value, dict):
        return params_value
    if pd.isna(params_value):
        return {}
    return ast.literal_eval(str(params_value))


def _build_regressor(model_name, params):
    normalized = str(model_name).strip().upper()
    model_map = {
        "RIDGE": Ridge,
        "LASSO": Lasso,
        "ELASTICNET": ElasticNet,
    }

    if normalized not in model_map:
        raise ValueError(f"Modelo não suportado no BEST_MODELS_BY_CLUSTER: {model_name}")

    return model_map[normalized](**params)


def train_predict_models_by_cluster(df_predict_by_cluster, best_models_csv_path, output_dir):
    best_models_df = pd.read_csv(best_models_csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trained_models = {}

    for _, row in best_models_df.iterrows():
        cluster_id = int(row["CLUSTER"])
        model_name = row["BEST_MODEL"]
        params = _parse_params(row["BEST_PARAMS"])

        if cluster_id not in df_predict_by_cluster:
            continue

        cluster_df = df_predict_by_cluster[cluster_id].copy()

        if "PROFIT_SCALED" not in cluster_df.columns:
            raise ValueError(f"Cluster {cluster_id} sem coluna alvo PROFIT_SCALED")

        y = cluster_df["PROFIT_SCALED"]
        X = cluster_df.drop(columns=["PROFIT_SCALED", "CLUSTER"], errors="ignore")

        regressor = _build_regressor(model_name, params)
        regressor.fit(X, y)

        model_file = output_path / f"cluster_{cluster_id}_{str(model_name).lower()}.joblib"
        joblib.dump(regressor, model_file)

        trained_models[cluster_id] = {
            "model_name": model_name,
            "model_path": str(model_file),
            "features": X.columns.tolist(),
        }

    return trained_models