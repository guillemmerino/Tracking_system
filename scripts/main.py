import csv
import os
import numpy as np
from asginacion_def import seleccionar_asignacion_definitiva
from scipy.optimize import linear_sum_assignment
from visualizar_tracking import visualizar_tracking
from lstm_track import predecir_lstm
from limpieza_kp import limpiar_keypoints_por_mapeo

BASE = os.getenv("DATA_DIR", "../csv")  # valor por defecto dentro del contenedor
CSV = os.path.join(BASE, "apilado.csv")
# Abrimos el archivo csv
ruta_csv = CSV
with open(ruta_csv, 'r') as f:
    lector = csv.reader(f)
    next(lector)  # Saltamos la cabecera
    datos = list(lector)

personas_anterior = []
personas_anterior_ID = []
next_id = 0
umbral = 0.1  # Ajusta según tus datos
frame_actual = None
frames_personas = []
personas_actual = []
personas_actual_ID = []
historial = {}
umbral_prediccion = 0.5
ids_invalidos = {}
dict_predicciones = {}
keypoints_mod = []
desaparecidos_greedy = {}
desaparecidos = {}
umbral_desaparecidos = 0.5
count = 0

for fila in datos:
    frame = int(fila[0])
    keypoints_csv = np.array([
        float(x) if x.strip() != '' else 0.0
        for x in fila[1:]
    ])
    keypoints_csv = limpiar_keypoints_por_mapeo(keypoints_csv)
    # Si cambia el frame, procesar asignaciones
    if frame != frame_actual and personas_actual:
        #print("Frame actual:", frame_actual)
        # HAY QUE HACER UNA LIMPIEZA DE KEYPOINTS PARA QUE NO INCLUYA LAS COORDENADAS 
        # QUE NO SE USAN EN LA LSTM (NI EN EL CODIGO)

        # -----------------------------------
        # 1) LLAMAMOS A ASIGNACION DEFINITIVA
        # -----------------------------------
        #print("Frame actual:", frame_actual)
        personas_actual_ID, personas_anterior_ID, desaparecidos_frame, next_id = seleccionar_asignacion_definitiva(
            personas_actual, personas_anterior, dict_predicciones, next_id,
            umbral, peso_pred_vs_prev=0.5, frame_actual=frame_actual
        )


        # -----------------------------------
        # 2) ACTUALIZACION IDs DESAPARECIDOS 
        # -----------------------------------
        # Comprobamos la lista de desaparecidos en este frame y actualizamos la lsita
        # de desaparecidos global
        for id_, keypoints in desaparecidos_frame.items():
            if id_ not in desaparecidos:
                desaparecidos[id_] = {"keypoints": keypoints, "frame_count" : 0}

        

        # -----------------------------------
        # 3) REIDENTIFICACIÓN IDs ANTIGUOS
        # -----------------------------------
        dict_predicciones = {}
        # Se toma la asignación por proximidad y se valora si hay un ID ya fijo
        for persona in personas_actual_ID:
            id_ = persona['id']
            keypoints_actuales = persona['keypoints']

            # Si este ID no existía en historial, podría ser "nuevo" => intenta recuperar
            if id_ is None or id_ not in historial:
                mejor_id = None
                mejor_dist = float('inf')
                for id_olvid, info in desaparecidos.items():
                    dist = np.linalg.norm(keypoints_actuales - info["keypoints"])
                    if dist < mejor_dist:
                        mejor_dist, mejor_id = dist, id_olvid

                if mejor_id is not None and mejor_dist < umbral_desaparecidos:
                    # RECUPERA EL ID ANTIGUO
                    persona['id'] = mejor_id
                    #print(f"Se recupera el ID {mejor_id} para la persona con ID previo {id_}")
                    # ya no está desaparecido
                    desaparecidos.pop(mejor_id, None)
                    

        # -------------------------------------------------
        # 4) ACTUALIZAMOS HISTORIAL
        # -------------------------------------------------
        for persona in personas_actual_ID:
            id_ = persona['id']            # <-- refrescado tras una posible reasignación
            keypoints = persona['keypoints']
            if id_ not in historial:
                historial[id_] = []
            if len(historial[id_]) > 0:
                last_valid = historial[id_][-1]
                keypoints = np.where(keypoints == 0, last_valid, keypoints)
            historial[id_].append(keypoints)
            persona['keypoints'] = keypoints  # Actualizamos los keypoints con el último válido
            if len(historial[id_]) > 22:
                historial[id_] = historial[id_][-20:]

        # -----------------------------------
        # 5)  COMPROBACIÓN DE IDS REPETIDOS 
        # -----------------------------------

        ids_en_frame = [p['id'] for p in personas_actual_ID]

        if len(ids_en_frame) != len(set(ids_en_frame)):
            print(f"ERROR: IDs repetidos en frame {frame_actual}: {ids_en_frame}")
            raise ValueError(f"IDs repetidos en frame {frame_actual}: {ids_en_frame}")
        frames_personas.append([dict(p) for p in personas_actual_ID])


        # -----------------------------------
        # 6)  ACTUALIZAMOS CONTADORES 
        # -----------------------------------

        desaparecidos_copy = desaparecidos.copy()
        for id_, value in desaparecidos_copy.items():
            value["frame_count"] += 1
            if value["frame_count"] > 20:
                # Si lleva más de 20 frames desaparecido
                #print(f"Se elimina el ID {id_} de los desaparecidos")
                desaparecidos.pop(id_, None)

        # ---------------------------------------------------
        # 7)  REALIZAMOS PREDICCIONES PARA EL SIGUIENTE FRAME 
        # ---------------------------------------------------
        # Por cada persona del frame actual
        for persona in personas_actual_ID:
            id_ = persona['id']
            # Si en su historial hay más de 20 frames, predecimos la siguiente posición            
            if len(historial[id_]) >= 20:
                secuencia = np.array(historial[id_][-20:])  # Últimos 20 frames
                
                #print("Secuencia shape:", secuencia.shape)
                prediccion = predecir_lstm(secuencia)
                # Si predecir_lstm devuelve el dict completo:
                if isinstance(prediccion, dict) and "predictions" in prediccion:
                    pred = np.array(prediccion["predictions"])
                else:
                    pred = np.array(prediccion)

                pred = np.squeeze(pred)
                if pred.ndim > 1:
                    pred = pred[0]                
                #print(f"Frame {frame} | ID {id_} | Predicción LSTM: {prediccion}")
                # Guardamos la predicción para ese ID en el dict_predicciones
                dict_predicciones[id_] = pred

        # El diccionario dict_predicciones es de la forma
        # {ID : 'keypoints'}

        personas_anterior = [dict(p) for p in personas_actual_ID]
        personas_actual = []


    personas_actual.append({'id': None, 'keypoints': keypoints_csv, 'frame': frame})
    frame_actual = frame

# Procesar el último frame
if personas_actual:
    print("Last frame")

    #print("Frame actual:", frame_actual)
    personas_def, personas_anterior, desaparecidos_frame, next_id = seleccionar_asignacion_definitiva(
        personas_actual, personas_anterior, dict_predicciones, next_id,
        umbral_geom=1.0, peso_pred_vs_prev=0.5, frame_actual=frame_actual
)    
    
    # -----------------------------------
    # 2) ACTUALIZACION IDs DESAPARECIDOS 
    # -----------------------------------
    # Comprobamos la lista de desaparecidos en este frame y actualizamos la lsita
    # de desaparecidos global
    for id_, keypoints in desaparecidos_greedy.items():
        if id_ not in desaparecidos:
            desaparecidos[id_] = {"keypoints": keypoints, "frame_count" : 1}
        else:
            # Si ya estaba, actualizamos los keypoints
            desaparecidos[id_]["frame_count"] += 1
            desaparecidos[id_]["keypoints"] = keypoints

        if desaparecidos[id_]["frame_count"] > 20:
            # Si lleva más de 20 frames desaparecido
            #print(f"Se elimina el ID {id_} de los desaparecidos")
            desaparecidos.pop(id_, None)

    # -----------------------------------
    # 3) REIDENTIFICACIÓN IDs ANTIGUOS
    # -----------------------------------
    dict_predicciones = {}
    # Se toma la asignación por proximidad y se valora si hay un ID ya fijo
    for persona in personas_actual_ID:
        id_ = persona['id']
        keypoints_actuales = persona['keypoints']

        # Si este ID no existía en historial, podría ser "nuevo" => intenta recuperar
        if id_ is None or id_ not in historial:
            mejor_id = None
            mejor_dist = float('inf')
            for id_olvid, info in desaparecidos.items():
                dist = np.linalg.norm(keypoints_actuales - info["keypoints"])
                if dist < mejor_dist:
                    mejor_dist, mejor_id = dist, id_olvid

            if mejor_id is not None and mejor_dist < umbral_desaparecidos:
                # RECUPERA EL ID ANTIGUO
                persona['id'] = mejor_id
                # ya no está desaparecido
                desaparecidos.pop(mejor_id, None)
                

    # -----------------------------------
    # 4) Historial (usa SIEMPRE el ID actual/corregido)
    # -----------------------------------

    for persona in personas_actual_ID:
        id_ = persona['id']            # <-- refrescado tras una posible reasignación
        kp = persona['keypoints']
        if id_ not in historial:
            historial[id_] = []
        if len(historial[id_]) > 0:
            last_valid = historial[id_][-1]
            kp = np.where(kp == 0, last_valid, kp)
        persona['keypoints'] = kp  # Actualizamos los keypoints con el último válido
        historial[id_].append(kp)
        if len(historial[id_]) > 20:
            historial[id_] = historial[id_][-20:]

    # 5) --- COMPROBACIÓN DE IDS REPETIDOS ---
    ids_en_frame = [p['id'] for p in personas_actual_ID]

    if len(ids_en_frame) != len(set(ids_en_frame)):
        print(f"ERROR: IDs repetidos en frame {frame_actual}: {ids_en_frame}")
        raise ValueError(f"IDs repetidos en frame {frame_actual}: {ids_en_frame}")
    frames_personas.append([dict(p) for p in personas_actual_ID])


visualizar_tracking(frames_personas, output_gif="tracking.gif")



''' # HAY QUE HACER UNA LIMPIEZA DE KEYPOINTS PARA QUE NO INCLUYA LAS COORDENADAS 
        # QUE NO SE USAN EN LA LSTM (NI EN EL CODIGO)

        # -----------------------------------
        # 1) LLAMAMOS A ASIGNACION GREEDY
        # -----------------------------------
        # TIENE EL PROBLEMA DE QUE SI UNA PERSONA DESAPARECE, O NO SE DETECTAN
        # GRAN PARTE DE SUS ARTICULACIONES, ASIGNA LOS IDs MAL (PUES SE TOMAN 
        # LOS VALORES ANTERIORES Y LAS DIFERENCIAS SE APROXIMAN A 0)

        personas_actual_ID_greedy, personas_anterior_ID_greedy, next_id, desaparecidos_greedy = asignar_ids_por_proximidad(
            personas_actual, personas_anterior, next_id, umbral
        )

        # -------------------------------------
        # 2) LLAMAMOS A LA ASINGACIÓN POR LSTM
        # -------------------------------------
        # Si ya tenemos una predicción
        if len(dict_predicciones) > 0:
            
            personas_actual_ID_lstm, dict_predicciones, desaparecidos_lstm, next_id = asignar_ids_por_lstm(
                personas_actual, personas_anterior, next_id, umbral_prediccion
            )

        # ------------------------------------------
        # 3) LLAMAMOS A LA ASINGACIÓN POR HUNGARIAN
        # ------------------------------------------
        # Si ya tenemos una predicción
            
        personas_actual_ID_hung, personas_anterior_ID_hung, desaparecidos_hung, next_id = asignar_ids_por_hungaro(
            personas_actual, personas_anterior, next_id, umbral_prediccion
        )


        # -------------------------------------------
        # 3) COMPARAMOS ASIGNACIONES Y DESAPARECIDOS
        # -------------------------------------------

        personas_actual_ID = personas_anterior_ID_greedy
        desaparecidos = desaparecidos_greedy
'''
