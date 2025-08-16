import os, requests
import numpy as np
from scipy.optimize import linear_sum_assignment

MODEL_URL = os.getenv("MODEL_URL", "http://lstm:8000")

def predecir_lstm(data_2d):
    """data_2d: lista de listas con shape (n_muestras, n_features) en el MISMO orden que entrenamiento."""
    data_2d = data_2d.tolist()
    r = requests.post(f"{MODEL_URL}/predict", json={"data": data_2d}, timeout=30)
    r.raise_for_status()
    body = r.json()
    if "error" in body:
        raise ValueError(body["error"])
    return body["predictions"]


def asignar_ids_por_lstm(personas_actual, dict_predicciones, next_id, umbral_prediccion=1.0):
    """
    Empareja personas del frame actual con predicciones LSTM usando el método Húngaro.
    
    Parámetros
    ----------
    personas_actual : list[dict]
        Cada dict debe tener al menos 'keypoints' (np.ndarray o lista de floats). El 'id' se asigna aquí.
    dict_predicciones : dict[int, np.ndarray]
        Mapea id_pred -> keypoints_predichos (vector/flatten con mismas dimensiones que 'keypoints').
    next_id : int
        Siguiente ID disponible para nuevas personas no emparejadas con predicción.
    umbral_prediccion : float
        Distancia máxima para considerar válida una asignación.

    Retorna
    -------
    asignadas : list[dict]
        Lista de personas_actual con el campo 'id' asignado.
    desaparecidos : dict[int, np.ndarray]
        Predicciones (por id_pred) que NO se asignaron a ninguna persona actual.
    next_id : int
        Siguiente ID actualizado tras asignar IDs nuevos a personas no emparejadas.
    """

    # Copia superficial para no modificar la lista original fuera (opcional)
    personas_actual = personas_actual.copy()
    
    ids_pred = list(dict_predicciones.keys())
    preds = np.array([dict_predicciones[i] for i in ids_pred]) if ids_pred else np.empty((0, 0))
    kp_actual = np.array([p['keypoints'] for p in personas_actual]) if personas_actual else np.empty((0, 0))

    #print ("keypoints reales:", kp_actual)
    #print ("Keypoints predicción LSTM:", preds)
    #print ("Distancia entre keypoints reales y predicción LSTM:", np.linalg.norm(kp_actual - preds))
    M = len(personas_actual)
    N = len(ids_pred)

    desaparecidos = {}

    # Casos borde
    if M == 0 and N == 0:
        return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos
    
    if M == 0 and N > 0:
        # No hay personas actuales -> todas las predicciones quedan como desaparecidas
        for pred in dict_predicciones.items():
            desaparecidos[pred["id"]] = pred["keypoints"]
        return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos
        
    if M > 0 and N == 0:
        # No hay predicciones -> todas las personas actuales reciben ID nuevo
        for p in personas_actual:
            p['id'] = next_id
            next_id += 1
        return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos

    # Matriz de distancias (M, N)
    # Asegúrate de que kp_actual y preds tengan la misma dimensión de features
    dist_matrix = np.linalg.norm(kp_actual[:, None, :] - preds[None, :, :], axis=2)
    #print("Matriz de distancias:", dist_matrix)
    # Umbral: penaliza con coste grande para que Húngaro no las elija
    BIG = 1e9
    cost = dist_matrix.copy()
    cost[cost >= umbral_prediccion] = BIG
    #print("Matriz de costes (con umbral aplicado):", cost)
    row_ind, col_ind = linear_sum_assignment(cost)

    # Emparejamientos válidos bajo umbral
    usados_pred = set()
    for i, j in zip(row_ind, col_ind):
        if dist_matrix[i, j] < umbral_prediccion:
            personas_actual[i]['id'] = ids_pred[j]
            usados_pred.add(j)

    # Personas no emparejadas -> IDs nuevos
    for i, p in enumerate(personas_actual):
        if 'id' not in p or p['id'] is None:
            p['id'] = next_id
            next_id += 1

    # Predicciones no usadas -> desaparecidos
    for j, pid in enumerate(ids_pred):
        if j not in usados_pred:
            desaparecidos[pid] = dict_predicciones[pid]

    return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos
