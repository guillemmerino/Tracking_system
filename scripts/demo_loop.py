# tracking_system/demo_loop.py
import numpy as np
from service_client import predict_next_pose

def main():
    K = 17  # por ejemplo, COCO
    kp = np.random.rand(K, 2).astype(np.float32) * 640  # puntos aleatorios
    pred = predict_next_pose(kp)
    print("Predicción recibida (shape):", pred.shape)
    print(pred[:3])  # muestra 3 puntos

if __name__ == "__main__":
    main()
