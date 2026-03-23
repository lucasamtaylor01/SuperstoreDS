# SuperstoreDS

## 📊 Descrição
Projeto de análise e modelagem com o dataset Superstore, focado em EDA, clusterização e previsão de lucro.


## ⚙️ Instalar dependências

   **Linux/macOS:**
   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   ```
   **Windows (PowerShell)**
   ```bash
   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
   ```

## 📂 Estrutura

```plaintext
.
├── 📂 data
│   ├── 📂 data_processed
│   │   ├── 📄 SUPERSTORE_MODELAGEM.csv
│   │   └── 📄 SUPERSTORE_TRATADO.csv
│   └── 📂 data_raw
│       └── 📄 SUPERSTORE.csv
├── 🐍 main.py
├── 📂 model
│   └── 📄 SUPERSTORE_CLUSTERING.csv
├── 📓 notebooks
│   ├── 📘 01_exploracao_inicial.ipynb
│   ├── 📘 02_outliers.ipynb
│   ├── 📘 03_kmeans.ipynb
│   └── 📘 04_temporal_series.ipynb
├── 📂 output
│   └── 📂 models_predict
│       ├── 🤖 arima_cluster_0.joblib
│       ├── 🤖 arima_cluster_1.joblib
│       ├── 🤖 arima_cluster_2.joblib
│       └── 🤖 kmeans_clustering.joblib
├── 📝 README.md
├── 📊 report
│   ├── 📂 notebook_01
│   ├── 📂 notebook_03
│   └── 📂 notebook_04
├── 📦 requirements.txt
└── 📂 src
    ├── ⚙️ build_model.py
    ├── 🔮 predict.py
    └── 🧰 utils.py
```

## 💾 DataSet

Disponível no [Github](https://raw.githubusercontent.com/WuCandice/Superstore-Sales-Analysis/refs/heads/main/dataset/Superstore%20Dataset.csv)
