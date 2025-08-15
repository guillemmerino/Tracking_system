import os, requests

MODEL_URL = os.getenv("MODEL_URL", "http://lstm:8000")

def predecir_lstm(data_2d):
    """data_2d: lista de listas con shape (n_muestras, n_features) en el MISMO orden que entrenamiento."""
    r = requests.post(f"{MODEL_URL}/predict", json={"data": data_2d}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise ValueError(body["error"])
    return body["predictions"]

