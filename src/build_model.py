from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def choose_best_k(df_metrics):
    df = df_metrics.copy()

    # normalizar tudo entre 0 e 1
    df["sil_norm"] = (df["silhouette"] - df["silhouette"].min()) / (df["silhouette"].max() - df["silhouette"].min())
    df["ch_norm"] = (df["calinski_harabasz"] - df["calinski_harabasz"].min()) / (df["calinski_harabasz"].max() - df["calinski_harabasz"].min())

    # inverter métricas que são "menor é melhor"
    df["db_norm"] = 1 - (df["davies_bouldin"] - df["davies_bouldin"].min()) / (df["davies_bouldin"].max() - df["davies_bouldin"].min())
    df["inertia_norm"] = 1 - (df["inertia"] - df["inertia"].min()) / (df["inertia"].max() - df["inertia"].min())

    # score final (média)
    df["score"] = df[["sil_norm", "ch_norm", "db_norm", "inertia_norm"]].mean(axis=1)

    # melhor k
    best_k = df["score"].idxmax()

    return best_k, df

def train_model(X, n_clusters=3):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    model.fit(X)
    return model

def save_model(model):
    joblib.dump(model, "model/model.joblib")