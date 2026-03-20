import joblib

def load_model(path="model/model.joblib"):
    """
    Carrega modelo salvo
    """
    return joblib.load(path)

def predict(model, X):
    """
    Gera predições (clusters)
    """
    return model.predict(X)