from pathlib import Path
import pandas as pd
from src.utils import dict_analise_temporal, preprocess, clean_data, data_clustering
from src.build_model import train_arima_by_cluster, train_kmeans
from src.predict import predict


# DEFINIÇÃO DE PATHS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'data_raw'
PROCESSED_DATA_DIR = DATA_DIR / 'data_processed'
DATA_PATH = RAW_DATA_DIR / 'SUPERSTORE.csv'
OUTPUT_DIR = BASE_DIR / 'output'
MODEL_DIR = BASE_DIR / 'model'

OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)


# TRATAMENTO DE DADOS INICIAL
df = pd.read_csv(DATA_PATH, sep=',')
df_tratado = clean_data(df)
df_tratado.to_csv(PROCESSED_DATA_DIR / 'SUPERSTORE_TRATADO.csv', index=False)

print('[1/4] Dados tratados exportados com sucesso.\n')


# TRATAMENTO DE DADOS PARA MODELAGEM
df_modelagem = preprocess(df)
df_modelagem.to_csv(PROCESSED_DATA_DIR / 'SUPERSTORE_MODELAGEM.csv', index=False)

print('[2/4] Dados para modelagem exportados com sucesso.\n')


# MODELO DE CLUSTERING (K-MEANS)
X_clustering, df_clustering = data_clustering(df_modelagem)
model_clustering = train_kmeans(
    X=X_clustering,
    k=3,
    random_state=0,
    output_path=OUTPUT_DIR / 'models_predict' / 'kmeans_clustering.joblib'
)
df_clustering['CLUSTER'] = predict(model_clustering, X_clustering)
df_clustering.to_csv(
    MODEL_DIR / 'SUPERSTORE_CLUSTERING.csv',
    index=False
)
print('[3/4] Modelo de clustering treinado e resultados exportados com sucesso.\n')

# MODELO DE SÉRIES TEMPORAIS POR CLUSTER
df_temporal_dict = dict_analise_temporal(df_clustering)
arima_results = train_arima_by_cluster(df_temporal_dict, output_dir=OUTPUT_DIR / 'models_predict')

print('[4/4] Modelos ARIMA treinados com sucesso.\n')
    
print('[!] Pipeline finalizado com sucesso.\n')
