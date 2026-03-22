from pathlib import Path
import pandas as pd
import joblib
from pmdarima import auto_arima
from sklearn.cluster import KMeans


def train_kmeans(X, k=3, random_state=0, output_path=None):
    # Treina um modelo K-Means com melhores parâmetro e salva modelo
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

def train_arima_by_cluster(df_temporal_dict, train_ratio=0.8, output_dir=None):

    # Treina um modelo ARIMA para cada cluster e salva os modelos
    
    arima_results = {}

    for i, data in df_temporal_dict.items():
        ts = data.set_index('ORDER_DATE')['PROFIT']
        ts = ts.asfreq('MS', fill_value=0)

        split_idx = int(train_ratio * len(ts))
        train = ts[:split_idx]
        test = ts[split_idx:]

        model = auto_arima(
            train,
            seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True
        )

        forecast = model.predict(n_periods=len(test))
        forecast_series = pd.Series(forecast, index=test.index)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            model_path = output_dir / f'arima_cluster_{i}.joblib'
            joblib.dump(model, model_path)

        arima_results[i] = {
            'train': train,
            'test': test,
            'forecast': forecast_series,
            'model': model
        }


    return arima_results
