import csv
import os
import numpy as np
from asginacion_def import seleccionar_asignacion_definitiva
from scipy.optimize import linear_sum_assignment
from visualizar_tracking import visualizar_tracking
from lstm_track import predecir_lstm
from limpieza_kp import limpiar_keypoints_por_mapeo
from service_client import calculate_score, esta_saltando

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
frames_personas = []
personas_actual = []
personas_actual_ID = []
historial = {}
ids_invalidos = {}
dict_predicciones = {}
desaparecidos = {}
personas_saltando = {}
recientes_evaluados = {}
hist_saltadores = {}
umbral_desaparecidos = 0.5
next_id = 0
umbral = 0.1  # Ajusta según tus datos
frame_actual = None
memoria_hist = 100
paciencia_saltador = {}


for i, fila in enumerate(datos):
    frame = int(fila[0])
    keypoints_csv = np.array([
        float(x) if x.strip() != '' else 0.0
        for x in fila[1:]
    ])
    keypoints_csv = limpiar_keypoints_por_mapeo(keypoints_csv)
    # Si cambia el frame, procesar asignaciones
    if (frame != frame_actual and personas_actual) or (i == len(datos) - 1 and personas_actual):
        #if frame_actual % 50 == 0:
        print("Frame actual:", frame_actual, next_id)
        # HAY QUE HACER UNA LIMPIEZA DE KEYPOINTS PARA QUE NO INCLUYA LAS COORDENADAS
        # QUE NO SE USAN EN LA LSTM (NI EN EL CODIGO)
        next_id_prev = next_id
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
                last_valid = historial[id_][-1]["keypoints"]
                # En caso que la persona este saltando, se rellena con la predicción LSTM
                if id_ in personas_saltando:
                    last_pred = dict_predicciones.get(id_, last_valid)
                    last_valid = last_pred
                keypoints = np.where(keypoints == 0, last_valid, keypoints)
            historial[id_].append({"frame": frame_actual, "keypoints": keypoints})
            # EN CASO QUE ESTE SALTANDO LA PERSONA SE PUEDE RELLENAR CON LSTM
            persona['keypoints'] = keypoints  # Actualizamos los keypoints con el último válido
            
            if len(historial[id_]) > memoria_hist:
                historial[id_] = historial[id_][-memoria_hist:]

        
        # Actualizamos tambien el historial de aquellas personas que estan saltando pero que, por lo que
        # sea, han desaparecido este frame
        for id_, frame_saltando in personas_saltando.items():
            if id_ in historial and historial[id_][-1]["frame"] != frame_actual:
                historial[id_].append({"frame": frame_actual, "keypoints": dict_predicciones.get(id_, np.zeros((17, 3)))})
                if id_ not in paciencia_saltador:
                    paciencia_saltador[id_] = frame_actual
            #print(f"Se añade al historial el ID {id_} con predicción LSTM")

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
        olvidados_ids = []
        desaparecidos_copy = desaparecidos.copy()
        for id_, value in desaparecidos_copy.items():
            value["frame_count"] += 1
            if value["frame_count"] > memoria_hist:
                # Si lleva más de 20 frames desaparecido
                #print(f"Se elimina el ID {id_} de los desaparecidos")
                print(f"Se olvida el ID {id_}")
                olvidados_ids.append(id_)
                desaparecidos.pop(id_, None)

                # Eliminamos el historial del olvidado
                if id_ in historial:
                    historial.pop(id_, None)
                
                if id_ in recientes_evaluados:
                    recientes_evaluados.pop(id_, None)
                
                if id_ in personas_saltando:
                    personas_saltando.pop(id_, None)

                if id_ in hist_saltadores:
                    hist_saltadores.pop(id_, None)

                if id_ in paciencia_saltador:
                    paciencia_saltador.pop(id_, None)

        # ---------------------------------------------------
        # 7)  REALIZAMOS PREDICCIONES PARA EL SIGUIENTE FRAME 
        # ---------------------------------------------------
        dict_predicciones = {}
        # Por cada persona del frame actual
        for persona in personas_actual_ID:
            id_ = persona['id']
            # Si en su historial hay más de 20 frames, predecimos la siguiente posición            
            if len(historial[id_]) >= 20:
                secuencia = np.array([h["keypoints"] for h in historial[id_][-20:]])  # Últimos 20 frames

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

                # Guardamos la predicción para ese ID en el dict_predicciones
                dict_predicciones[id_] = pred


        # Tambien realizamos predicciones de aquellas personas saltando que no esten en el frame
        for id_, frame_saltando in personas_saltando.items():
            if id_ not in dict_predicciones:
                secuencia = np.array([h["keypoints"] for h in historial[id_][-20:]])  # Últimos 20 frames
                prediccion = predecir_lstm(secuencia)
                if isinstance(prediccion, dict) and "predictions" in prediccion:
                    pred = np.array(prediccion["predictions"])
                else:
                    pred = np.array(prediccion)

                pred = np.squeeze(pred)
                if pred.ndim > 1:
                    pred = pred[0]

                dict_predicciones[id_] = pred

        # Si hay algun ID asignado mayor a next_id_prev, significa que hemos asignado un nuevo ID
        if any(p['id'] > next_id_prev for p in personas_actual_ID):
            # Se retorna el next_ID
            pass

        # Si no se ha asignado un nuevo ID, se mantiene el anterior
        else:
            next_id = next_id_prev
            #print(f"Se mantiene el ID anterior: {next_id}")

        # ---------------------------------------------------
        # 8)  VEMOS SI LA PERSONA ESTA SALTANDO 
        # ---------------------------------------------------
        #print ("PASO (8)")
        #print ("Personas en el frame actual:", len(personas_actual_ID))
        # OJO, SOLO SE EVALUAN LAS PERSONAS QUE APARECEN EN ESE FRAME. SI ALGUIEN SE GUARDA COMO SALTANDO
        # Y DESAPARECE EN ESE FRAME, HAYQ UE VER QUE HACER
        for persona in personas_actual_ID:
            id_ = persona['id']

                # Buscar si ya está en personas_saltando y obtener el último frame en que se evaluó
            debe_evaluar = False
            if id_ in recientes_evaluados and (frame_actual - recientes_evaluados[id_] >= 50):
                recientes_evaluados[id_] = frame_actual
                debe_evaluar = True
            else:
                if id_ not in recientes_evaluados:
                    recientes_evaluados[id_] = frame_actual
                    debe_evaluar = True

                else:
                    debe_evaluar = False

            if id_ is not None and id_ in historial and len(historial[id_]) >= memoria_hist and debe_evaluar:
                jumping = False
                solo_keypoints = [h["keypoints"] for h in historial[id_][-memoria_hist:]]
                secuencia = np.array(solo_keypoints)
                jumping = esta_saltando(secuencia)
                print ("Evaluando salto para ID:", id_, frame_actual, jumping)
                if jumping:
                    if id_ in personas_saltando:
                        print ("sigue saltando")
                    
                    # Eliminamos el candidato a dejar de saltar
                    if id_ in paciencia_saltador:
                        paciencia_saltador.pop(id_, None)
                        print("Eliminamos de paciencia al ID:", id_)
                    personas_saltando[id_] = frame_actual

                else:
                    # Si la persona no esta saltando pero estaba en el diccionario, la eliminamos
                    if id_ in personas_saltando:
                        personas_saltando.pop(id_, None)
                        # Añadimos al ID que dejo de saltar para esperar a si vuelve a saltar o no
                        paciencia_saltador[id_] = frame_actual
                        print("Dejó de saltar")
            # HAY QUE HACER UN TRATAMIENTO ESPECIAL DE LAS PERSONAS SALTANDO, PUES NO PUEDEN TENER
            # KEYPOINTS IGUAL A 0. 

            # De las personas saltando, accedemos a las últimas posiciones de sus articulaciones
            # guardadas en el historial

        eliminados = []
        print ("Paciencia saltadores:", paciencia_saltador)
        for id_, frame_paciencia in paciencia_saltador.items():
           # Revisamos el diccionario de paciencia
            if frame_actual - frame_paciencia > 90:
                # Eliminamos el saltador
                eliminados.append(id_)
                if id_ in personas_saltando:
                    personas_saltando.pop(id_, None)
                hist_saltadores.pop(id_, None)
                print("Eliminamos el ID de saltador:", id_)
        
        for id_ in eliminados:
            paciencia_saltador.pop(id_, None)

        for id_, frame_saltando in personas_saltando.items():
            # Accedemos al historial de esta persona:
            hist_saltadores[id_] = historial[id_][-memoria_hist:]
        # De las personas que estan saltando, mandamos sus keypoints para que se procesen en el servicio
        # del cálculo de notas
        print ("Historial saltadores: ", hist_saltadores.keys())
        for id_, hist in hist_saltadores.items():
            solo_keypoints = [h["keypoints"] for h in hist]
            score = calculate_score(np.array(solo_keypoints))
            print ("Nota para el ID", id_, ":", score)
        # El diccionario dict_predicciones es de la forma
        # {ID : 'keypoints'}

        personas_anterior = [dict(p) for p in personas_actual_ID]
        personas_actual = []


    personas_actual.append({'id': None, 'keypoints': keypoints_csv, 'frame': frame})
    frame_actual = frame

visualizar_tracking(frames_personas)