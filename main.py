from pathlib import Path
import pandas as pd

from src.utils import preprocess, scale_data
from src.build_model import evaluate_k, choose_best_k, train_model, save_model
from src.predict import predict


# caminho independente de SO
BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "data" / "data_raw" / "SUPERSTORE_DATASET.csv"

df = pd.read_csv(data_path, sep=",")

# 2. preprocessamento
df = preprocess(df)

# 3. selecionar features numéricas
X = df.select_dtypes(include="number")

# 4. scaling (importante pro KMeans)
X, scaler = scale_data(X, X.columns)

# 5. avaliar k
df_metrics = evaluate_k(X)

# 6. escolher melhor k
k, df_metrics = choose_best_k(df_metrics)

# 7. treinar modelo final
model = train_model(X, n_clusters=k)

# 8. prever clusters
df["CLUSTER"] = predict(model, X)

# 9. salvar resultados
df.to_csv("output/predicted.csv", index=False)

# 10. salvar modelo
save_model(model)