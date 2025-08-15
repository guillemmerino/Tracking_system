import csv
import numpy as np



def asignar_ids_por_proximidad(personas_actual, personas_anterior, next_id, umbral=1.0):
    """
    Asigna IDs a personas_actual comparando con personas_anterior por proximidad.
    Si algún valor de keypoints en personas_actual es 0, se reemplaza por el último valor válido del mismo ID.
    Retorna la lista actualizada, la nueva lista de personas_anterior y el nuevo next_id.
    """

    desaparecidos = {}
     # Filtra personas con todos los keypoints a 0

    # Si no había anteriores, asigna IDs nuevos y termina
    if not personas_anterior:
        for p in personas_actual:
            p['id'] = next_id
            next_id += 1
        return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos


    keypoints_actual = np.array([p['keypoints'] for p in personas_actual])
    keypoints_anterior = np.array([p['keypoints'] for p in personas_anterior])
    ids_anterior = [p['id'] for p in personas_anterior]


    M = len(personas_actual)
    N = len(personas_anterior)
    if M == 0:
        # no hay actuales -> todos los anteriores "desaparecen"
        for p in personas_anterior:
            desaparecidos[p['id']] = p['keypoints']
        return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos


    # Relleno por-par: si un valor actual es 0, usa el del candidato anterior j
    # Resultado: (M, N, D)
    actual_filled_3d = np.where(
        keypoints_actual[:, None, :] == 0,
        keypoints_anterior[None, :, :],
        keypoints_actual[:, None, :]
    )

    # Distancia euclídea por-par tras el relleno
    diff = actual_filled_3d - keypoints_anterior[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)  # (M, N)
    

    asignaciones = [None] * len(personas_actual)
    usados_actual = set()
    usados_anterior = set()

    pares = [
        (i, j, dist_matrix[i, j])
        for i in range(len(personas_actual))
        for j in range(len(personas_anterior))
    ]

    pares.sort(key=lambda x: x[2])

    for i, j, dist in pares:
        if dist < umbral and i not in usados_actual and j not in usados_anterior:
            #print(f"Distancia {dist} dentro de umbral {umbral}, se asigna ID")
            asignaciones[i] = ids_anterior[j]
            usados_actual.add(i)
            usados_anterior.add(j)

    # Se asigna el mismo ID de antes o uno nuevo
    for idx, persona in enumerate(personas_actual):
        if asignaciones[idx] is not None:
            persona['id'] = asignaciones[idx]
        else:
            persona['id'] = next_id
            #print(f"Aparece nueva persona: {next_id}")
            next_id += 1

    # Diccionario para acceder rápidamente a los keypoints anteriores por ID
    keypoints_prev_map = {}
    if personas_anterior:
        for p in personas_anterior:
            keypoints_prev_map[p['id']] = p['keypoints']

    # Rellenar valores 0 en keypoints con el último valor válido del mismo ID
    for persona in personas_actual:
        if persona['id'] is not None and persona['id'] in keypoints_prev_map:
            last_valid = keypoints_prev_map[persona['id']]
            persona['keypoints'] = np.where(persona['keypoints'] == 0, last_valid, persona['keypoints'])

    # Personas desaparecidas: aquellas de personas_anterior cuyo índice no está en usados_anterior
    for j, p in enumerate(personas_anterior):
        if j not in usados_anterior:
            #print(f"Persona desaparecida: {p['id']}")
            desaparecidos[p['id']] = p['keypoints']

    # Retornar la lista actualizada, la nueva lista de personas_anterior, el next_id actualizado y el diccionario de desaparecidos
    return personas_actual, [dict(p) for p in personas_actual], next_id, desaparecidos

