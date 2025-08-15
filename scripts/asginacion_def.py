import numpy as np
from copy import deepcopy
from knn_track import asignar_ids_por_proximidad
from hungarian_track import asignar_ids_por_hungaro
from lstm_track import asignar_ids_por_lstm

def _map_prev(personas_anterior):
    """id -> keypoints del frame anterior"""
    return {p['id']: p['keypoints'] for p in personas_anterior}

def _dist(a, b):
    if a is None or b is None:
        return np.inf
    return float(np.linalg.norm(a - b))

def _dist_to_prev(kp_actual, id_candidato, prev_map):
    """Distancia del actual (ya con relleno si lo hizo el asignador) a su 'id' anterior."""
    if id_candidato in prev_map:
        return _dist(kp_actual, prev_map[id_candidato])
    return np.inf

def _dist_to_pred(kp_actual, id_candidato, dict_predicciones):
    """Distancia del actual a la predicción LSTM de ese id (si existe)."""
    if id_candidato in dict_predicciones:
        return _dist(kp_actual, np.array(dict_predicciones[id_candidato]))
    return np.inf

def seleccionar_asignacion_definitiva(
    personas_actual,           # list[{'id': None, 'keypoints': np.ndarray, 'frame': int}]
    personas_anterior,         # list[{'id': int, 'keypoints': np.ndarray, ...}]
    dict_predicciones,         # dict[id:int -> keypoints_pred: np.ndarray]
    next_id,
    umbral_geom,           # igual que tus otros umbrales
    peso_pred_vs_prev=0.7,      # 0 => solo pasado, 1 => solo predicción; 0.5 mezcla
    frame_actual=None
):
    """
    Devuelve:
      personas_def, personas_anterior_nuevo, desaparecidos_def, next_id
    """
    personas_filtradas = []
    for persona in personas_actual:
        if np.all(persona['keypoints'] == 0):
            # Añade a desaparecidos con id None (o puedes manejarlo como prefieras)
            print("Persona con todo 0 identificada")
        else:
            personas_filtradas.append(persona)
    personas_actual = personas_filtradas

    # Copias para no mutar lo original al llamar a los asignadores
    P0 = deepcopy(personas_actual)
    P1 = deepcopy(personas_actual)
    P2 = deepcopy(personas_actual)

    if frame_actual > 85 and frame_actual < 95:
        print("Personas_actuales:", len(personas_actual))


    # 1) Propuestas
    # greedy / proximidad
    prop_greedy, prev_greedy, next_id, desc_greedy = asignar_ids_por_proximidad(P0, personas_anterior, next_id, umbral=umbral_geom)  # 
    # húngaro (contra el frame anterior)
    prop_hung,   prev_hung, next_id, desc_hung = asignar_ids_por_hungaro(P1, personas_anterior, next_id, umbral=umbral_geom)      # 

    if frame_actual > 85 and frame_actual < 95:
        print("Propuestas en frame actual:", prop_greedy, prop_hung)
    # lstm (si hay predicciones)
    hay_lstm = bool(dict_predicciones)
    if hay_lstm:
        prop_lstm, prev_lstm, next_id, desc_lstm = asignar_ids_por_lstm(P2, dict_predicciones, next_id, umbral_prediccion=umbral_geom)  # 
    else:
        prop_lstm, prev_lstm, next_id, desc_lstm = None, None, next_id, {}

    # 2) Mayoría + desempate por mejor coste
    prev_map = _map_prev(personas_anterior)
    personas_def = deepcopy(personas_actual)  # mantengo el orden original
    usados_ids = set()

    # Para cada persona_def, buscamos el match más cercano en cada propuesta por keypoints
    for i, persona in enumerate(personas_def):
        kp_i = persona['keypoints']
        candidatos = {}
        # Buscar el id más cercano en cada propuesta
        for nombre, propuesta in [('greedy', prop_greedy), ('hung', prop_hung)]:
            mejor_j = None
            mejor_dist = float('inf')
            for p in propuesta:
                dist = np.linalg.norm(kp_i - p['keypoints'])
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_j = p
            if mejor_j is not None and mejor_dist < umbral_geom:
                candidatos[nombre] = mejor_j['id']
        #print("Candidatos para persona", i, ":", candidatos, "frame", frame_actual)
        if hay_lstm and prop_lstm is not None:
            mejor_j = None
            mejor_dist = float('inf')
            for p in prop_lstm:
                dist = np.linalg.norm(kp_i - p['keypoints'])
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_j = p
            if mejor_j is not None and mejor_dist < umbral_geom:
                candidatos['lstm'] = mejor_j['id']

        # (a) mayoría exacta
        conteo = {}
        for m, pid in candidatos.items():
            conteo[pid] = conteo.get(pid, 0) + 1
        pid_mayoria = None
        for pid, c in conteo.items():
            if pid is not None and c >= 2:
                pid_mayoria = pid
                if frame_actual > 85 and frame_actual < 95:
                    print("Mayoría encontrada en frame", frame_actual, ":", pid_mayoria)
                break

        # (b) si hay mayoría y no está usado, me lo quedo
        elegido = None
        if pid_mayoria is not None and pid_mayoria not in usados_ids:
            elegido = pid_mayoria
        else:
            # (c) si no hay mayoría, puntuamos cada candidato por coste:
            mejores = []
            for metodo, pid in candidatos.items():
                if pid is None:
                    continue
                d_prev = _dist_to_prev(kp_i, pid, prev_map)
                d_pred = _dist_to_pred(kp_i, pid, dict_predicciones)
                if not hay_lstm or np.isinf(d_pred):
                    coste = d_prev
                else:
                    coste = (1 - peso_pred_vs_prev) * d_prev + peso_pred_vs_prev * d_pred
                mejores.append((coste, pid, metodo))
            if mejores:
                mejores.sort(key=lambda x: x[0])
                for coste, pid, _ in mejores:
                    if coste < umbral_geom and pid not in usados_ids:
                        elegido = pid
                        break
        if elegido is None or elegido in usados_ids:
            elegido = next_id
            next_id += 1
        persona['id'] = elegido
        persona['keypoints'] = kp_i
        usados_ids.add(elegido)

    # 3) unifico desaparecidos (un “OR” de las fuentes)
    desaparecidos_def = {}
    for blob in (desc_greedy, desc_hung, desc_lstm):
        for k, v in blob.items():
            desaparecidos_def[k] = v
    if desaparecidos_def:
        print("Desaparecidos en este frame:", desaparecidos_def.keys())

    # 4) nuevo “personas_anterior” = copia del definitivo
    personas_anterior_nuevo = [dict(p) for p in personas_def]

    return personas_def, personas_anterior_nuevo, desaparecidos_def, next_id
