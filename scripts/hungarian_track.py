import numpy as np
from scipy.optimize import linear_sum_assignment

def asignar_ids_por_hungaro(personas_actual, personas_anterior, next_id, umbral=1.0):
    """
    Asigna IDs usando el método Húngaro (minimizando distancias) con:
      - Relleno: si un keypoint actual es 0, se usa el del candidato anterior (por par).
      - Umbral: asignaciones con coste >= umbral se consideran NO válidas.
      - Desaparecidos: anteriores no emparejados.
    Retorna: (personas_actual_actualizadas, nueva_personas_anterior, next_id, desaparecidos)
    """
    desaparecidos = {}

    # 0) Filtrar entradas "todas a 0"
    personas_filtradas = []
    for persona in personas_actual:
        if np.all(persona['keypoints'] == 0):
            # puedes loggear si quieres
            pass
        else:
            personas_filtradas.append(persona)
    personas_actual = personas_filtradas

    # Si no había anteriores, asigna IDs nuevos y termina
    if not personas_anterior:
        for p in personas_actual:
            p['id'] = next_id
            next_id += 1
        return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos

    # 1) Preparar matrices
    keypoints_actual = np.array([p['keypoints'] for p in personas_actual])        # (M, D)
    keypoints_anterior = np.array([p['keypoints'] for p in personas_anterior])    # (N, D)
    ids_anterior = [p['id'] for p in personas_anterior]

    M = len(personas_actual)
    N = len(personas_anterior)
    if M == 0:
        # no hay actuales -> todos los anteriores "desaparecen"
        for p in personas_anterior:
            desaparecidos[p['id']] = p['keypoints']
        return personas_actual, [], next_id, desaparecidos

    # 2) Relleno por-par (actual 0 -> valor del candidato anterior j)
    # actual_filled_3d: (M, N, D)
    actual_filled_3d = np.where(
        keypoints_actual[:, None, :] == 0,
        keypoints_anterior[None, :, :],
        keypoints_actual[:, None, :]
    )

    # 3) Distancias por-par -> matriz de costes (M, N)
    diff = actual_filled_3d - keypoints_anterior[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)

    # 4) Método Húngaro
    # Para manejar el umbral: ponemos un coste grande a los emparejamientos que superen el umbral,
    # pero OJO: linear_sum_assignment siempre produce M o N emparejamientos.
    # Luego filtramos los que exceden el umbral.
    BIG = 1e9
    cost = dist_matrix.copy()
    cost[cost >= umbral] = BIG

    # Hungarian
    row_ind, col_ind = linear_sum_assignment(cost)  # indices i (actual) -> j (anterior)

    # 5) Construir asignaciones respetando el umbral
    asignaciones = [None] * M
    usados_anterior = set()
    for i, j in zip(row_ind, col_ind):
        if dist_matrix[i, j] < umbral:
            asignaciones[i] = ids_anterior[j]
            usados_anterior.add(j)
        # si no, se queda None y se tratará como nueva persona

    # 6) Asignar IDs (reusar o nuevos)
    for i, persona in enumerate(personas_actual):
        if asignaciones[i] is not None:
            persona['id'] = asignaciones[i]
        else:
            persona['id'] = next_id
            next_id += 1

    # 7) Relleno de keypoints =0 con último válido del mismo ID
    keypoints_prev_map = {p['id']: p['keypoints'] for p in personas_anterior}
    for p in personas_actual:
        pid = p.get('id')
        if pid in keypoints_prev_map:
            last_valid = keypoints_prev_map[pid]
            p['keypoints'] = np.where(p['keypoints'] == 0, last_valid, p['keypoints'])

    # 8) Desaparecidos: anteriores no usados
    for j, p in enumerate(personas_anterior):
        if j not in usados_anterior:
            desaparecidos[p['id']] = p['keypoints']

    # Retorno
    return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos
