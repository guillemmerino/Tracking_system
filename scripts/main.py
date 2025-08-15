import csv
import os
import numpy as np
from knn_track import asignar_ids_por_proximidad
from hungarian_track import asignar_ids_por_hungaro
from lstm_track import predecir_lstm
from visualizar_tracking import visualizar_tracking

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
umbral = 1.0  # Ajusta según tus datos
frame_actual = None
frames_personas = []
personas_actual = []
personas_actual_ID = []
historial = {}
#modelo_lstm = cargar_modelo_lstm()
umbral_prediccion = 0.5
ids_invalidos = {}
dict_predicciones = {}
keypoints_mod = []
desaparecidos_frame = {}
desaparecidos = {}
umbral_desaparecidos = 0.5
count = 0

for fila in datos:
    frame = int(fila[0])
    keypoints_csv = np.array([
        float(x) if x.strip() != '' else 0.0
        for x in fila[1:]
    ])
    # Si cambia el frame, procesar asignaciones
    if frame != frame_actual and personas_actual:


        # -----------------------------------
        # 1) LLAMAMOS A ASIGNACION POR PROXIMIDAD
        # -----------------------------------
        # TIENE EL PROBLEMA DE QUE SI UNA PERSONA DESAPARECE, O NO SE DETECTAN
        # GRAN PARTE DE SUS ARTICULACIONES, ASIGNA LOS IDs MAL (PUES SE TOMAN 
        # LOS VALORES ANTERIORES Y LAS DIFERENCIAS SE APROXIMAN A 0)

        personas_actual_ID, personas_anterior_ID, next_id, desaparecidos_frame = asignar_ids_por_hungaro(
            personas_actual, personas_anterior, next_id, umbral
        )

        # Se devuelve una lista de diccionarios (ahora personas_actual = personas_anterior) el cual da 
        # [{ID , 'keypoints', 'frame'},...]
 

        # -----------------------------------
        # 2) ACTUALIZACION IDs DESAPARECIDOS 
        # -----------------------------------
        # Comprobamos la lista de desaparecidos en este frame y actualizamos la lsita
        # de desaparecidos global
        for id_, keypoints in desaparecidos_frame.items():
            if id_ not in desaparecidos:
                desaparecidos[id_] = {"keypoints": keypoints, "frame_count" : 0}

        

        # -----------------------------------
        # COMPROBAMOS LA PREDICCIÓN SI HAY
        # -----------------------------------
        # Si ya tenemos una predicción
        if len(dict_predicciones) > 0:
            # Iteramos sobre las personas del frame actual, vemos si el punto con 
            # ID ya asignado tiene una predicción, y vemos la distancia entre ellos
            for persona in personas_actual_ID:
                if persona['id'] in dict_predicciones:
                    keypoints = dict_predicciones[persona['id']]
                    distancia = np.linalg.norm(keypoints - persona['keypoints'])

                    # Damos por valida o no la asignacion por knn
                    if distancia < umbral_prediccion:
                        print(f"Se valida la prediccion del ID {persona['id']}, distancia {distancia}")

                    else:
                        print(f"Se invalida la prediccion del ID {persona['id']}, distancia {distancia}")
                        ids_invalidos[persona['id']] = persona['keypoints']


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
                    
        #if count >80 and count < 100:
        #    print("Desaparecidos:", desaparecidos.keys())

        #    print("Personas frame: ", [p["id"] for p in personas_actual_ID])
        #count += 1
        # -----------------------------------
        # 4) Historial (usa SIEMPRE el ID actual/corregido)
        # -----------------------------------
    
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
            if len(historial[id_]) > 20:
                historial[id_] = historial[id_][-20:]

        # -----------------------------------
        # 5)  COMPROBACIÓN DE IDS REPETIDOS 
        # -----------------------------------

        ids_en_frame = [p['id'] for p in personas_actual_ID]

        if len(ids_en_frame) != len(set(ids_en_frame)):
            print(f"ERROR: IDs repetidos en frame {frame_actual}: {ids_en_frame}")
            raise ValueError(f"IDs repetidos en frame {frame_actual}: {ids_en_frame}")
        frames_personas.append([dict(p) for p in personas_actual_ID])

        desaparecidos_copy = desaparecidos.copy()
        for id_, value in desaparecidos_copy.items():
            value["frame_count"] += 1
            if value["frame_count"] > 20:
                # Si lleva más de 20 frames desaparecido
                #print(f"Se elimina el ID {id_} de los desaparecidos")
                desaparecidos.pop(id_, None)



        # El diccionario dict_predicciones es de la forma
        # {ID : 'keypoints'}

        personas_anterior = [dict(p) for p in personas_actual_ID]
        #if frame_actual < 5:
        #    print(f"Frame {frame_actual}, personas actual ID: {[p['keypoints'] for p in personas_actual_ID]}")
        #    print(f"Frame {frame_actual}, personas anterior: {[p['keypoints'] for p in personas_anterior]}")
        personas_actual = []


    personas_actual.append({'id': None, 'keypoints': keypoints_csv, 'frame': frame})
    frame_actual = frame

# Procesar el último frame
if personas_actual:
    personas_actual_ID, personas_anterior_ID, next_id, desaparecidos_frame = asignar_ids_por_proximidad(
            personas_actual, personas_anterior, next_id, umbral
        )

    # -----------------------------------
    # 2) ACTUALIZACION IDs DESAPARECIDOS 
    # -----------------------------------
    # Comprobamos la lista de desaparecidos en este frame y actualizamos la lsita
    # de desaparecidos global
    for id_, keypoints in desaparecidos_frame.items():
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
'''        # -----------------------------------
        # LLAMAMOS A ASIGNACION POR LSTM
        # -----------------------------------

            # Si el historial es mayor a 20, predice la siguiente posición
            if len(historial[id_]) > 20:
                secuencia = np.array(historial[id_][-20:])  # Últimos 20 frames
                prediccion = predecir_lstm(modelo_lstm, secuencia)
                print(f"Frame {frame} | ID {id_} | Predicción LSTM: {prediccion}")
            else:
                prediccion = []

            # Creamos un diccionario con las predicciones de cada ID para el siguiente
            # frame
            dict_predicciones[id_] = prediccion'''
