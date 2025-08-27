import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

# ----------------------------
# Utilidades de geometría
# ----------------------------

def l2(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float('inf')
    return float(np.linalg.norm(a - b))

# Obtenemos las boundary boxes para solaparlas después
def keypoints_to_box(kp: np.ndarray) -> Tuple[float, float, float, float]:
    """Crea un bbox (xmin, ymin, xmax, ymax) a partir de keypoints.
    Si los keypoints tienen ceros, los ignora para el bbox.
    """
    kp = np.asarray(kp)
    paso = 2  # SIN SCORES!!
    xs = kp[0::paso] if kp.ndim == 1 else kp[:, 0]
    ys = kp[1::paso] if kp.ndim == 1 else kp[:, 1]
    # Filtra puntos en cero
    m = (xs != 0) & (ys != 0)
    if not np.any(m):
        return (0.0, 0.0, 0.0, 0.0)
    xs = xs[m]
    ys = ys[m]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    return (xmin, ymin, xmax, ymax)

# Intersección sobre Unión (IoU)
def iou_from_boxes(a: Tuple[float, float, float, float],
                   b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)

# Centro de masa simple para Kalman
def keypoints_center(kp: np.ndarray) -> np.ndarray:
    """Centro aproximado del sujeto a partir de keypoints (media de puntos válidos)."""
    kp = np.asarray(kp)
    if kp.ndim == 1:
        paso = 2
        xs = kp[0::paso]
        ys = kp[1::paso]
    else:
        xs = kp[:, 0]
        ys = kp[:, 1]
    m = (xs != 0) & (ys != 0)
    if not np.any(m):
        return np.array([np.nan, np.nan], dtype=float)
    return np.array([float(xs[m].mean()), float(ys[m].mean())], dtype=float)


# ----------------------------
# Kalman Filter (CV en 2D)
# Estado: [x, y, vx, vy]
# Medida: [x, y]
# ----------------------------

class KalmanCV2D:
    def __init__(self, q: float = 1e-3, r_pos: float = 1e-4):
        """
        q: potencia de ruido de proceso (aceleración blanca). Más alto => más permisivo.
        r_pos: varianza de la medición en píxeles (puedes escalarla por tamaño de bbox si quieres).
        """
        self.q_base = float(q)
        self.r_pos = float(r_pos)
        self.x = None  # estado (4,1)
        self.P = None  # covarianza (4,4)
        # Matriz de proyeccion de estado oculto x= (x,y, vel_x, vel_y) a medición z= (x,y)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float) 
        # Ruido de medición
        self.R = np.eye(2, dtype=float) * self.r_pos

    def _F_Q(self, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        dt = float(max(1e-6, dt))
        # Matriz de transición de estado x = x_prev + vel * dt (velocidades no cambian)
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        q = self.q_base
        # Incertidumbres atribuídas a la aceleración (contempladas en P)
        Q = q * np.array([[dt4/4,    0,    dt3/2, 0],
                          [0,    dt4/4,    0,    dt3/2],
                          [dt3/2,  0,      dt2,  0],
                          [0,    dt3/2,    0,    dt2]], dtype=float)
        return F, Q
    

    # Estados iniciales
    def init_from_measurement(self, z_xy: np.ndarray):
        x, y = float(z_xy[0]), float(z_xy[1])
        
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        # Matriz solo diagonal para iniciar
        self.P = np.diag([1e-3, 1e-3, 1e-2, 1e-2])  # incertidumbre alta al inicio

    # Se predice la posición con su incertidumbre asociada
    def predict(self, dt: float):
        if self.x is None:
            raise RuntimeError("Kalman no inicializado")
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        # Incertidumbre
        self.P = F @ self.P @ F.T + Q

    def innovation_cov(self) -> np.ndarray:
        """S = H P H^T + R"""
        return self.H @ self.P @ self.H.T + self.R

    # Distancia umbral
    def mahalanobis2(self, z_xy: np.ndarray) -> float:
        z = np.array(z_xy, dtype=float).reshape(2, 1)
        zhat = (self.H @ self.x)
        y = z - zhat
        S = self.innovation_cov()
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        d2 = float((y.T @ Sinv @ y)[0, 0])
        return d2

    # Se corrige la incertidumbre dada la medición real (corrección de la medición)
    def update(self, z_xy: np.ndarray):
        z = np.array(z_xy, dtype=float).reshape(2, 1)
        H = self.H
        # Innovación (cuanto de distinta es la medición a la predicción)
        # S = covarianza de la innovación
        S = self.innovation_cov()
        try:
            # Peso que decide cuanto moverse a la predicción (P grande, K grande)
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)
        y = z - (H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T

        # piso de covarianza (evita S minúscula ⇒ d² gigantes)
        Pmin = np.diag([1e-5, 1e-5, 5e-4, 5e-4])  # ajusta a tu escena
        diagP = np.diag(self.P).copy()
        np.fill_diagonal(self.P, np.maximum(diagP, np.diag(Pmin)))

    def predicted_xy(self) -> np.ndarray:
        return np.array([(self.H @ self.x)[0, 0], (self.H @ self.x)[1, 0]], dtype=float)


# ----------------------------
# Asignadores existentes (tus módulos)
# ----------------------------
from knn_track import asignar_ids_por_proximidad
from hungarian_track import asignar_ids_por_hungaro
from lstm_track import asignar_ids_por_lstm


# ----------------------------
# Coste enriquecido + GATING
# ----------------------------

def coste_enriquecido(kp_actual: np.ndarray,
                      kp_ref_prev: Optional[np.ndarray],
                      kp_ref_pred: Optional[np.ndarray],
                      bbox_actual: Tuple[float, float, float, float],
                      bbox_pred: Optional[Tuple[float, float, float, float]] = None,
                      conf: Optional[float] = None,
                      w_prev: float = 0.5,
                      w_pred: float = 1.0,
                      w_iou: float = 0.5,
                      w_conf: float = 0.2) -> float:
    """
    Combina varias señales en un único coste. Si alguna no está disponible, su contribución se omite.
    """
    C = 0.0
    w_sum = 0.0

    if kp_ref_prev is not None:
        C += w_prev * l2(kp_actual, kp_ref_prev); w_sum += w_prev

    if kp_ref_pred is not None:
        C += w_pred * l2(kp_actual, kp_ref_pred); w_sum += w_pred

    if bbox_pred is not None:
        iou = iou_from_boxes(bbox_pred, bbox_actual)
        C += w_iou * (1.0 - iou); w_sum += w_iou

    if conf is not None:
        # Asumimos conf en [0,1]. Si está en otra escala, normalízala antes.
        C += w_conf * (1.0 - float(conf)); w_sum += w_conf

    if w_sum == 0.0:
        return l2(kp_actual, kp_ref_prev) if kp_ref_prev is not None else float('inf')
    return C / w_sum


def seleccionar_asignacion_definitiva(
    personas_actual: List[dict],           # cada dict: {'id': None, 'keypoints': np.ndarray, 'frame': int, 'conf': opt}
    personas_anterior: List[dict],
    dict_predicciones: Dict[int, np.ndarray],   # id -> keypoints pred LSTM (opcional)
    kalmans: Dict[int, KalmanCV2D],             # id -> estado Kalman (persistente)
    next_id: int,
    umbral_geom: float,
    dt_seconds: float = 1/30.0,                # si no tienes PTS reales, usa 1/fps
    gating_dist_max_per_sec: float = 150.0,    # píxeles/seg: máximo desplazamiento plausible
    gating_maha_thresh: float = 9.21,          # chi^2 k=2 al 99%
    pesos_coste: Tuple[float, float, float, float] = (0.5, 1.0, 0.5, 0.2),
    frame_actual: Optional[int] = None
):
    """
    Devuelve: personas_def, personas_anterior_nuevo, desaparecidos_def, next_id, kalmans
    - Añade Kalman por track para predicción, gating y coste (Mahalanobis opcional si lo integras).
    - Incluye gating por distancia máxima dependiente de dt y por Mahalanobis (si hay Kalman).
    - Coste enriquecido: mezcla d_prev, d_pred, IoU y confianza (si están disponibles).
    """

    w_prev, w_pred, w_iou, w_conf = pesos_coste

    # Filtra personas con todos ceros en kp
    personas_filtradas = []
    for persona in personas_actual:
        kp = np.asarray(persona['keypoints'])
        if np.all(kp == 0):
            continue
        personas_filtradas.append(persona)
    personas_actual = personas_filtradas

    # Copias para no mutar
    P0 = deepcopy(personas_actual)
    P1 = deepcopy(personas_actual)
    P2 = deepcopy(personas_actual)

    # 1) Propuestas base
    prop_greedy, prev_greedy, next_id, desc_greedy = asignar_ids_por_proximidad(P0, personas_anterior, next_id, umbral=umbral_geom)
    prop_hung,   prev_hung,   next_id, desc_hung   = asignar_ids_por_hungaro(P1, personas_anterior, next_id, umbral=umbral_geom)

    hay_lstm = bool(dict_predicciones)
    if hay_lstm:
        prop_lstm, prev_lstm, next_id, desc_lstm = asignar_ids_por_lstm(P2, dict_predicciones, next_id, umbral_prediccion=0.1)
    else:
        prop_lstm, prev_lstm, next_id, desc_lstm = None, None, next_id, {}

    # Mapa id -> kp anterior
    prev_map = {p['id']: p['keypoints'] for p in personas_anterior}

    # 2) Inicializa/Predice Kalman por cada track previo
    for p in personas_anterior:
        tid = p['id']
        kp_prev = p['keypoints']
        c_prev = keypoints_center(kp_prev)
        if tid not in kalmans:
            kf = KalmanCV2D(q=1e-3, r_pos=1e-4)
            if not (np.isnan(c_prev).any()):
                kf.init_from_measurement(c_prev)
                kalmans[tid] = kf
        else:
            # predict al tiempo actual
            try:
                kalmans[tid].predict(dt_seconds)
            except RuntimeError:
                # si por lo que sea no estaba inicializado
                if not (np.isnan(c_prev).any()):
                    kalmans[tid].init_from_measurement(c_prev)

    # 3) Selección definitiva por mayoría + coste enriquecido + GATING
    personas_def = deepcopy(personas_actual)
    usados_ids = set()

    # Gating (filtrado) por desplazamiento máximo dependiente de dt
    dist_max = gating_dist_max_per_sec * max(1e-6, dt_seconds)

    desaparecidos_def = {}

    for i, persona in enumerate(personas_def):
        kp_i = np.asarray(persona['keypoints'])
        bbox_i = keypoints_to_box(kp_i)
        conf_i = persona.get('conf', None)

        candidatos: Dict[str, int] = {}
        # toma el mejor id de cada propuesta si pasa gating básico
        for nombre, propuesta in [('greedy', prop_greedy), ('hung', prop_hung)]:
            mejor = None
            mejor_dist = float('inf')
            for p in propuesta:
                d = l2(kp_i, p['keypoints'])
                if d < mejor_dist:
                    mejor_dist = d; mejor = p
            if mejor is None:
                continue
            pid = mejor['id']

            # Gating: distancia máxima vs predicción Kalman si existe
            ok = True
            if pid in kalmans:
                pred_xy = kalmans[pid].predicted_xy()
                cen_i = keypoints_center(kp_i)
                if not (np.isnan(cen_i).any()):
                    if l2(cen_i, pred_xy) > dist_max:
                        ok = False
                    else:
                        # opcional: gating por Mahalanobis
                        d2 = kalmans[pid].mahalanobis2(cen_i)
                        if d2 > gating_maha_thresh:
                            ok = False
                        else:
                            print(f"Gating Mahalanobis OK para ID {pid} (d2={d2:.12f})")
            else:
                # si no hay Kalman, usa distancia al kp previo si existe
                if pid in prev_map:
                    if l2(kp_i, prev_map[pid]) > dist_max:
                        ok = False

            if ok:
                # pasa gating
                if mejor_dist < umbral_geom:
                    candidatos[nombre] = pid
                else:
                    print(f"{nombre}: mejor distancia {mejor_dist:.2f} > umbral {umbral_geom:.4f} para ID {pid}")

        # LSTM como fuente adicional
        if hay_lstm and prop_lstm is not None:
            mejor = None
            mejor_dist = float('inf')
            for p in prop_lstm:
                d = l2(kp_i, p['keypoints'])
                if d < mejor_dist:
                    mejor_dist = d; mejor = p
            if mejor is not None and mejor_dist < umbral_geom:
                pid = mejor['id']
                # gating con Kalman/prev también
                ok = True
                if pid in kalmans:
                    pred_xy = kalmans[pid].predicted_xy()
                    cen_i = keypoints_center(kp_i)
                    if not (np.isnan(cen_i).any()):
                        if l2(cen_i, pred_xy) > dist_max:
                            ok = False
                        else:
                            d2 = kalmans[pid].mahalanobis2(cen_i)
                            if d2 > gating_maha_thresh:
                                ok = False
                elif pid in prev_map:
                    if l2(kp_i, prev_map[pid]) > dist_max:
                        ok = False
                if ok:
                    candidatos['lstm'] = pid

        # (a) mayoría
        conteo: Dict[int, int] = {}
        for _, pid in candidatos.items():
            if pid is None:
                continue
            conteo[pid] = conteo.get(pid, 0) + 1
        pid_mayoria = None
        for pid, c in conteo.items():
            if pid is not None and c >= 2:
                pid_mayoria = pid
                break

        elegido = None
        if pid_mayoria is not None and pid_mayoria not in usados_ids:
            elegido = pid_mayoria
        else:
            # (b) sin mayoría: calcula coste enriquecido por cada candidato
            mejores: List[Tuple[float, int, str]] = []
            for metodo, pid in candidatos.items():
                if pid is None:
                    continue
                kp_prev = prev_map.get(pid, None)
                kp_pred = dict_predicciones.get(pid, None) if hay_lstm else None

                # bbox pred del track: del kp_prev o de una pseudo-predicción (aquí usamos kp_prev)
                bbox_pred = keypoints_to_box(kp_prev) if kp_prev is not None else None

                coste = coste_enriquecido(
                    kp_actual=kp_i,
                    kp_ref_prev=kp_prev,
                    kp_ref_pred=kp_pred,
                    bbox_actual=bbox_i,
                    bbox_pred=bbox_pred,
                    conf=conf_i,
                    w_prev=w_prev,
                    w_pred=w_pred,
                    w_iou=w_iou,
                    w_conf=w_conf,
                )
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
        usados_ids.add(elegido)

        # Actualiza/Inicializa Kalman con la medición actual
        cen_i = keypoints_center(kp_i)
        if elegido not in kalmans:
            kf = KalmanCV2D(q=1e-3, r_pos=1e-4)
            if not (np.isnan(cen_i).any()):
                kf.init_from_measurement(cen_i)
                kalmans[elegido] = kf
        else:
            if not (np.isnan(cen_i).any()):
                kalmans[elegido].update(cen_i)

    # Unifica desaparecidos (OR de las fuentes)
    for blob in (desc_greedy, desc_hung, desc_lstm):
        for k, v in blob.items():
            desaparecidos_def[k] = v

    personas_anterior_nuevo = [dict(p) for p in personas_def]
    return personas_def, personas_anterior_nuevo, desaparecidos_def, next_id, kalmans
