# tracking_system/service_client.py
import os, requests, numpy as np

MODEL_URL = os.getenv("JUMPING_SCORE_URL", "http://jumping_score:9000")

def predict_next_pose(kp: np.ndarray) -> np.ndarray:
    """
    kp: (K,2) ndarray en coords de imagen
    """
    payload = {"keypoints": kp.tolist()}
    r = requests.post(f"{MODEL_URL}/predict", json=payload, timeout=3.0)
    r.raise_for_status()
    out = r.json()["pred_keypoints"]
    return np.array(out, dtype=np.float32)


def calculate_score(kp : np.ndarray):
    """
    Se pasan los keypoints. El servicio detecta: 
    1) separa los saltos
    2) Identifica cuando un elemento es distinto a un bote simple
    3) Devuelve la puntuación.
    """
    kp = np.asarray(kp, dtype=np.float32)
    payload = {"data": kp.tolist()}
    r = requests.post(f"{MODEL_URL}/score", json=payload, timeout=3.0)
    r.raise_for_status()
    body = r.json()
    scores = body.get("score", [])

    # Acepta float, lista o lista de listas y la aplana a floats
    if isinstance(scores, (int, float)):
        return [float(scores)]
    out = []
    def _flat(xs):
        for e in xs:
            if isinstance(e, (list, tuple)):
                _flat(e)
            elif e is not None:
                out.append(float(e))
    _flat(scores)
    return out

def esta_saltando(kp : np.ndarray):
    """
    Se pasan los keypoints. El servicio detecta: 
    1) Maximos y minimos
    2) Si superan el umbral, devuelve True, sino False
    """
    payload = {"keypoints": kp.tolist()}
    r = requests.post(f"{MODEL_URL}/isjumping", json=payload, timeout=3.0)
    r.raise_for_status()
    return r.json()["is_jumping"]

