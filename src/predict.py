import joblib
import pandas as pd

def load_model(path):
    return joblib.load(path)


def predict(model, X):
    return model.predict(X)


def predict_arima(model, periods, index=None):
    forecast = model.predict(n_periods=periods)

    if index is not None:
        return pd.Series(forecast, index=index)

    return forecast