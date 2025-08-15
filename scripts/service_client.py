# tracking_system/service_client.py
import os, requests, numpy as np

MODEL_URL = os.getenv("MODEL_URL", "http://lstm:8000")

def predict_next_pose(kp: np.ndarray) -> np.ndarray:
    """
    kp: (K,2) ndarray en coords de imagen
    """
    payload = {"keypoints": kp.tolist()}
    r = requests.post(f"{MODEL_URL}/predict", json=payload, timeout=3.0)
    r.raise_for_status()
    out = r.json()["pred_keypoints"]
    return np.array(out, dtype=np.float32)
