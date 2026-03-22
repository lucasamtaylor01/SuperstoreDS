from pathlib import Path
import joblib
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge, ElasticNet


def train_predict_models_by_cluster(dfs_clusters, output_dir):
    
    MODEL_CONFIG = {
        0: ("LASSO", {"alpha": 0.001}),
        1: ("RIDGE", {"alpha": 1}),
        2: ("ELASTICNET", {"alpha": 0.01, "l1_ratio": 0.35}),
    }

    MODEL_MAP = {
        "LASSO": Lasso,
        "RIDGE": Ridge,
        "ELASTICNET": ElasticNet
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trained_models = {}

    for cluster, df_cluster in dfs_clusters.items():

        model_name, params = MODEL_CONFIG[cluster]

        X = df_cluster.drop(columns=['PROFIT_SCALED', 'CLUSTER'])
        y = df_cluster['PROFIT_SCALED']

        model = MODEL_MAP[model_name](**params)
        model.fit(X, y)

        model_path = output_dir / f'cluster_{cluster}_{model_name.lower()}.joblib'
        joblib.dump(model, model_path)

        trained_models[cluster] = model

    return trained_models



def train_kmeans(X, k=3, random_state=0, output_path=None):
    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10
    )
    
    model.fit(X)
    
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, output_path)
    
    return model