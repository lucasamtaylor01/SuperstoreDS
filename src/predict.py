import joblib


def load_model(path='model/model.joblib'):
    return joblib.load(path)


def predict(model, X):
    return model.predict(X)