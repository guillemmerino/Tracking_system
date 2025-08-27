import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, List, Optional

# ==========================================================
# Utilidades de geometría / KP helpers
# ==========================================================

def l2(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return float('inf')
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def split_xy(kp):
    """Acepta (J,2)/(J,3) o plano 2J/3J. Devuelve xs, ys (1D)."""
    kp = np.asarray(kp, dtype=float)
    if kp.ndim == 2 and kp.shape[1] >= 2:
        xs, ys = kp[:, 0], kp[:, 1]
    else:
        step = 3 if (kp.size % 3 == 0) else 2
        xs, ys = kp[0::step], kp[1::step]
    return xs, ys


def keypoints_to_box(kp: np.ndarray) -> Tuple[float, float, float, float]:
    xs, ys = split_xy(kp)
    m = (xs != 0) & (ys != 0)
    if not np.any(m):
        return (0.0, 0.0, 0.0, 0.0)
    xs, ys = xs[m], ys[m]
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())
    return (xmin, ymin, xmax, ymax)


def bbox_center_from_kp(kp: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = keypoints_to_box(kp)
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)


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


# ==========================================================
# Kalman Filter 2D (x,y,vx,vy) con forma de Joseph y piso de P
# ==========================================================

class KalmanCV2D:
    def __init__(self,
                 q: float = 1e-3,          # ruido de proceso (normalizado)
                 r_pos: float = 1e-4,      # varianza medición (σ≈0.01)
                 P0_pos: float = 1e-3,
                 P0_vel: float = 1e-2,
                 Pmin_pos: float = 1e-5,
                 Pmin_vel: float = 5e-4):
        self.q_base = float(q)
        self.r_pos = float(r_pos)
        self.P0_pos = float(P0_pos)
        self.P0_vel = float(P0_vel)
        self.Pmin_pos = float(Pmin_pos)
        self.Pmin_vel = float(Pmin_vel)

        self.x = None   # (4,1)
        self.P = None   # (4,4)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2, dtype=float) * self.r_pos

    def _F_Q(self, dt: float):
        dt = float(max(1e-6, dt))
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
        dt2, dt3, dt4 = dt*dt, dt*dt*dt, dt*dt*dt*dt
        q = self.q_base
        Q = q * np.array([[dt4/4, 0,      dt3/2, 0],
                          [0,      dt4/4, 0,      dt3/2],
                          [dt3/2, 0,      dt2,   0],
                          [0,      dt3/2, 0,      dt2]], dtype=float)
        return F, Q

    def init_from_measurement(self, z_xy: np.ndarray):
        x, y = float(z_xy[0]), float(z_xy[1])
        self.x = np.array([[x], [y], [0.0], [0.0]], dtype=float)
        self.P = np.diag([self.P0_pos, self.P0_pos, self.P0_vel, self.P0_vel]).astype(float)

    def predict(self, dt: float):
        if self.x is None:
            raise RuntimeError("Kalman no inicializado")
        F, Q = self._F_Q(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def innovation_cov(self) -> np.ndarray:
        return self.H @ self.P @ self.H.T + self.R

    def mahalanobis2(self, z_xy: np.ndarray) -> float:
        z = np.array(z_xy, dtype=float).reshape(2, 1)
        zhat = self.H @ self.x
        y = z - zhat
        S = self.innovation_cov()
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            Sinv = np.linalg.pinv(S)
        return float((y.T @ Sinv @ y)[0, 0])

    def update(self, z_xy: np.ndarray):
        z = np.array(z_xy, dtype=float).reshape(2, 1)
        H = self.H
        S = self.innovation_cov()
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            S_inv = np.linalg.pinv(S)
        K = self.P @ H.T @ S_inv
        y = z - (H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(4)
        # Joseph para estabilidad numérica
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
        # Piso de covarianza (evita sobreconfianza extrema)
        diagP = np.diag(self.P).copy()
        floors = np.array([self.Pmin_pos, self.Pmin_pos, self.Pmin_vel, self.Pmin_vel], dtype=float)
        np.fill_diagonal(self.P, np.maximum(diagP, floors))

    def predicted_xy(self) -> np.ndarray:
        return np.array([(self.H @ self.x)[0, 0], (self.H @ self.x)[1, 0]], dtype=float)


def retune_kf_for_normalized(kf: KalmanCV2D,
                             q: float = 1e-3,
                             r_pos: float = 1e-4,
                             P_pos: float = 1e-3,
                             P_vel: float = 1e-2,
                             trace_cap: float = 1.0):
    kf.q_base = float(q)
    kf.R = np.eye(2, dtype=float) * float(r_pos)
    if kf.P is None or np.trace(kf.P) > trace_cap:
        kf.P = np.diag([P_pos, P_pos, P_vel, P_vel]).astype(float)


# ==========================================================
# Coste enriquecido (KP, IoU, conf, Mahalanobis opcional)
# ==========================================================

def coste_enriquecido(
    kp_actual: np.ndarray,
    kp_ref_prev: Optional[np.ndarray],
    kp_ref_pred: Optional[np.ndarray],
    bbox_actual: Tuple[float, float, float, float],
    bbox_pred: Optional[Tuple[float, float, float, float]] = None,
    conf: Optional[float] = None,
    d2_maha: Optional[float] = None,
    pred_xy_kf: Optional[np.ndarray] = None,
    w_prev: float = 0.4,
    w_kf: float = 1.0,
    w_pred: float = 0.8,
    w_iou: float = 0.5,
    w_conf: float = 0.2,
    w_maha: float = 0.5,
    umbral_geom: Optional[float] = None,
    mejor_dist_kp: Optional[float] = None,
    tau_lstm=0.02,
) -> float:
    C, wsum = 0.0, 0.0
    if kp_ref_prev is not None:
        C += w_prev * l2(kp_actual, kp_ref_prev); wsum += w_prev
    if kp_ref_pred is not None:
        # LSTM en suave y dinámico según error de centro
        cen = bbox_center_from_kp(kp_actual)
        cen_lstm = bbox_center_from_kp(kp_ref_pred)
        e = float(np.linalg.norm(cen - cen_lstm))
        w_lstm_eff = w_pred * np.exp(-(e / tau_lstm) ** 2)
        C += w_lstm_eff * l2(kp_actual, kp_ref_pred); wsum += w_lstm_eff
    if pred_xy_kf is not None:
        cen = bbox_center_from_kp(kp_actual)
        C += w_kf * float(np.linalg.norm(cen - pred_xy_kf)); wsum += w_kf
    if bbox_pred is not None:
        iou = iou_from_boxes(bbox_pred, bbox_actual)
        C += w_iou * (1.0 - iou); wsum += w_iou
    if conf is not None:
        C += w_conf * (1.0 - float(conf)); wsum += w_conf
    if d2_maha is not None:
        C += w_maha * float(d2_maha); wsum += w_maha
    # Penalización suave por umbral geométrico (hinge)
    if umbral_geom is not None and mejor_dist_kp is not None:
        over = max(0.0, (mejor_dist_kp - umbral_geom) / max(1e-6, umbral_geom))
        C += 0.3 * over; wsum += 0.3
    return C / max(1e-6, wsum)


# ==========================================================
# Asignación con anti-hijack, cuarentena y re-ID segura
# ==========================================================

# (Opcional) importa tus asignadores si los sigues usando
try:
    from knn_track import asignar_ids_por_proximidad
    from hungarian_track import asignar_ids_por_hungaro
    from lstm_track import asignar_ids_por_lstm
    HAVE_EXTERNAL = True
except Exception:
    HAVE_EXTERNAL = False

CHI2_99  = 9.210340371976184  # df=2
CHI2_999 = 13.815510557964274 # df=2


def seleccionar_asignacion_definitiva(
    personas_actual: List[dict],           # [{'id': None, 'keypoints': np.ndarray, 'conf': opt}, ...]
    personas_anterior: List[dict],
    dict_predicciones: Dict[int, np.ndarray],   # id -> keypoints pred LSTM (opcional)
    kalmans: Dict[int, KalmanCV2D],             # id -> KF persistente
    track_meta: Dict[int, dict],                # pid -> metadatos (last_seen_frame, lost_time, just_coasted, quarantine_until, nis_ema)
    next_id: int,
    umbral_geom: float,
    dt_seconds: float = 1/30.0,
    v_max_per_s: float = 1.2,                  # normalizado
    gating_maha_hard: float = CHI2_999,        # descarte duro
    gating_maha_soft: float = CHI2_99,         # marca coasting
    pesos_coste: Tuple[float, float, float, float, float, float] = (1.0, 0.2, 0.1, 0.6, 0.1, 0.8),  # w_prev, w_pred, w_iou, w_conf, w_maha
    T_lost_seconds: float = 1.0,
    quarantine_seconds: float = 0.25,
    frame_actual: Optional[int] = None,
):
    """
    Devuelve: personas_def, personas_anterior_nuevo, desaparecidos_def, next_id, kalmans, track_meta
    - Gating: velocidad máx por Δt + Mahalanobis (soft/hard)
    - Anti-hijack: cuarentena temporal y mutual-nearest validation
    - Re-ID: posterior al match principal, estricta (Mahalanobis + IoU)
    - Caso "todos desaparecen": todos quedan en coasting y en cuarentena; se mantienen vivos hasta T_lost
    """

    w_kf, w_prev, w_pred, w_iou, w_conf, w_maha = pesos_coste

    # Reloj simple acumulado en metadatos
    track_meta['_clock'] = track_meta.get('_clock', 0.0) + float(dt_seconds)
    t_now = track_meta['_clock']

    # Filtro de personas con KP vacíos
    personas_actual = [p for p in personas_actual if not np.all(np.asarray(p['keypoints']) == 0)]

    # Predicción de KFs existentes + retune (normalizado)
    for p in personas_anterior:
        pid = p['id']
        if pid not in kalmans:
            # Inicializa on-demand al final cuando se asigne
            continue
        retune_kf_for_normalized(kalmans[pid])
        try:
            kalmans[pid].predict(dt_seconds)
        except RuntimeError:
            # si no estaba inicializado, lo haremos al asignar
            pass

    # Propuestas externas (si existen) — usamos solo para sugerir candidatos
    prop_greedy = prop_hung = prop_lstm = []
    if HAVE_EXTERNAL:
        P0 = deepcopy(personas_actual)
        P1 = deepcopy(personas_actual)
        P2 = deepcopy(personas_actual)
        prop_greedy, prev_greedy, next_id, _ = asignar_ids_por_proximidad(P0, personas_anterior, next_id, umbral=umbral_geom)
        prop_hung,   prev_hung,   next_id, _ = asignar_ids_por_hungaro(P1, personas_anterior, next_id, umbral=umbral_geom)
        if dict_predicciones:
            prop_lstm, prev_lstm, next_id, _ = asignar_ids_por_lstm(P2, dict_predicciones, next_id, umbral_prediccion=0.1)
    # Mapa previo
    prev_map = {p['id']: p['keypoints'] for p in personas_anterior}

    # --------- Construcción de candidatos con GATING + anti-hijack ---------
    dist_max = v_max_per_s * max(1e-6, dt_seconds)
    candidates: List[Tuple[int, int, float, float]] = []  # (pid, det_idx, cost, d2)

    # Precalcula centros/bboxes actuales
    det_centers = [bbox_center_from_kp(p['keypoints']) for p in personas_actual]
    det_bboxes  = [keypoints_to_box(p['keypoints']) for p in personas_actual]

    # Fuente de candidatos: unimos IDs sugeridos por las propuestas externas y los previos
    suggested_ids_per_det = []
    for i, persona in enumerate(personas_actual):
        kp_i = np.asarray(persona['keypoints'])
        # recoge mejores de cada propuesta (si existen)
        ids = set()
        for prop in (prop_greedy or []):
            # elegimos el id del más cercano de la propuesta a esta detección
            pid = prop.get('id', None)
            if pid is not None:
                ids.add(pid)
        for prop in (prop_hung or []):
            pid = prop.get('id', None)
            if pid is not None:
                ids.add(pid)
        for prop in (prop_lstm or []):
            pid = prop.get('id', None)
            if pid is not None:
                ids.add(pid)
        # añade todos los previos también (para robustez) — si es muy denso, puedes limitar por distancia al centro
        ids.update(prev_map.keys())
        suggested_ids_per_det.append(list(ids))

    def antihijack_ok(pid: int, det_idx: int) -> bool:
        m = track_meta.get(pid, {})
        cen_i = det_centers[det_idx]
        if pid in kalmans:
            d2 = kalmans[pid].mahalanobis2(cen_i)
        else:
            d2 = 0.0
        # Cuarentena: si el track acaba de desaparecer, solo permitir si es muy claro
        if t_now < m.get('quarantine_until', 0.0):
            if d2 > 5.99:  # 95% más estricto
                return False
        return True

    for i, persona in enumerate(personas_actual):
        kp_i = np.asarray(persona['keypoints'])
        cen_i = det_centers[i]
        bb_i  = det_bboxes[i]
        conf_i = persona.get('conf', None)

        for pid in suggested_ids_per_det[i]:
            # Gating geométrico por centro
            ok = True
            if pid in kalmans:
                pred_xy = kalmans[pid].predicted_xy()
                if l2(cen_i, pred_xy) > dist_max:
                    ok = False
                else:
                    d2 = kalmans[pid].mahalanobis2(cen_i)
                    if d2 > gating_maha_hard:
                        ok = False
                    elif d2 > gating_maha_soft:
                        # marca coasting; no aceptes directo, pero no mates el track
                        meta = track_meta.setdefault(pid, {})
                        meta['just_coasted'] = True
                        ok = False
                    # anti-hijack
                    if ok and not antihijack_ok(pid, i):
                        ok = False
            else:
                # sin Kalman: fallback al kp previo si existe
                if pid in prev_map:
                    if l2(kp_i, prev_map[pid]) > umbral_geom:
                        ok = False
                d2 = None

            if not ok:
                continue

            # Coste enriquecido (suave)
            kp_prev = prev_map.get(pid, None)
            kp_pred = dict_predicciones.get(pid, None) if dict_predicciones else None
            pred_xy_kf = kalmans[pid].predicted_xy() if pid in kalmans else None
            bb_pred = keypoints_to_box(kp_prev) if kp_prev is not None else None
            mejor_dist_kp = l2(kp_i, kp_prev) if kp_prev is not None else None
            d2_maha = kalmans[pid].mahalanobis2(cen_i) if pid in kalmans else None
            cost = coste_enriquecido(
                kp_actual=kp_i,
                kp_ref_prev=kp_prev,
                kp_ref_pred=kp_pred,
                bbox_actual=bb_i,
                bbox_pred=bb_pred,
                conf=conf_i,
                d2_maha=d2_maha,
                pred_xy_kf=pred_xy_kf,
                w_kf=w_kf, w_prev=w_prev, w_pred=w_pred, w_iou=w_iou, w_conf=w_conf, w_maha=w_maha,
                umbral_geom=umbral_geom, mejor_dist_kp=mejor_dist_kp,
            )
            candidates.append((pid, i, cost, d2 if pid in kalmans else 0.0))

    # Si no hay candidatos (p.ej., todos fuera de gate), podremos hacer re-ID o nacer nuevos

    # --------- Mutual-nearest validation / selección de matches ---------
    # agrupa por pid y por det idx para encontrar mejores
    best_det_for_track: Dict[int, Tuple[int, float]] = {}
    best_track_for_det: Dict[int, Tuple[int, float]] = {}
    for pid, i, cost, _ in candidates:
        if (pid not in best_det_for_track) or (cost < best_det_for_track[pid][1]):
            best_det_for_track[pid] = (i, cost)
        if (i not in best_track_for_det) or (cost < best_track_for_det[i][1]):
            best_track_for_det[i] = (pid, cost)

    assigned_det: Dict[int, int] = {}   # det i -> pid
    used_tracks: set = set()

    # 1) Acepta los mutuos
    for pid, (i, c) in best_det_for_track.items():
        pid2, c2 = best_track_for_det.get(i, (None, None))
        if pid2 == pid:
            assigned_det[i] = pid
            used_tracks.add(pid)

    # 2) Reglas extra: si no es mutuo, acepta solo si coste es muy bueno (d2 bajo + IoU decente)
    for pid, (i, c) in best_det_for_track.items():
        if i in assigned_det:
            continue
        if pid in used_tracks:
            continue
        cen_i = det_centers[i]
        d2 = kalmans[pid].mahalanobis2(cen_i) if pid in kalmans else 0.0
        iou_ok = True
        kp_prev = prev_map.get(pid, None)
        if kp_prev is not None:
            iou_ok = iou_from_boxes(keypoints_to_box(kp_prev), det_bboxes[i]) > 0.2
        if d2 <= 5.99 and iou_ok:
            assigned_det[i] = pid
            used_tracks.add(pid)

    # --------- Construye salida y estados ---------
    personas_def = deepcopy(personas_actual)
    ids_elegidos = set()

    # Inicializa y/o actualiza KFs con las mediciones aceptadas
    for i, persona in enumerate(personas_def):
        if i in assigned_det:
            pid = assigned_det[i]
            kp_i = np.asarray(persona['keypoints'])
            cen_i = det_centers[i]
            # asignar ID
            persona['id'] = pid
            ids_elegidos.add(pid)
            # init/update KF
            if pid not in kalmans:
                kf = KalmanCV2D()
                kf.init_from_measurement(cen_i)
                kalmans[pid] = kf
            else:
                kalmans[pid].update(cen_i)
            # meta
            m = track_meta.setdefault(pid, {})
            m['last_seen_frame'] = frame_actual
            m['lost_time'] = 0.0
            m['just_coasted'] = False
            m['quarantine_until'] = 0.0
        else:
            # Sin asignación todavía; lo resolveremos con re-ID o nacimiento
            persona['id'] = None

    # IDs previos / desaparecidos en este frame (para anti-hijack y re-ID)
    ids_previos = {p['id'] for p in personas_anterior}
    ids_faltan = ids_previos - ids_elegidos

    # Marca coasting + cuarentena para los que faltan
    for pid in ids_faltan:
        m = track_meta.setdefault(pid, {})
        m['just_coasted'] = True
        m['lost_time'] = m.get('lost_time', 0.0) + dt_seconds
        m['quarantine_until'] = max(m.get('quarantine_until', 0.0), t_now + quarantine_seconds)

    # --------- Re-ID desde desaparecidos (post-proceso estricto) ---------
    # Intenta asignar IDs faltantes a detecciones aún libres
    dets_libres = [i for i, p in enumerate(personas_def) if p['id'] is None]
    for i in dets_libres:
        cen_i = det_centers[i]
        bb_i  = det_bboxes[i]
        best_pid, best_score = None, 1e9
        for pid in ids_faltan:
            if pid not in kalmans:
                continue
            # si en cuarentena, exige gate más estricto
            if t_now < track_meta.get(pid, {}).get('quarantine_until', 0.0):
                if kalmans[pid].mahalanobis2(cen_i) > 5.99:
                    continue
            d2 = kalmans[pid].mahalanobis2(cen_i)
            kp_prev = prev_map.get(pid, None)
            bb_prev = keypoints_to_box(kp_prev) if kp_prev is not None else (0, 0, 0, 0)
            iou = iou_from_boxes(bb_prev, bb_i)
            score = d2 + (1.0 - iou)
            if score < best_score:
                best_score, best_pid = score, pid
        if best_pid is not None and best_score < 3.0:
            # aceptar re-ID
            personas_def[i]['id'] = best_pid
            ids_elegidos.add(best_pid)
            ids_faltan.discard(best_pid)
            kalmans[best_pid].update(cen_i)
            m = track_meta.setdefault(best_pid, {})
            m['last_seen_frame'] = frame_actual
            m['lost_time'] = 0.0
            m['just_coasted'] = False
            m['quarantine_until'] = 0.0

    # --------- Nacimiento de nuevos IDs para detecciones aún libres ---------
    for i, persona in enumerate(personas_def):
        if persona['id'] is None:
            persona['id'] = next_id
            # inicializa KF
            cen_i = det_centers[i]
            kf = KalmanCV2D()
            kf.init_from_measurement(cen_i)
            kalmans[next_id] = kf
            # meta
            m = track_meta.setdefault(next_id, {})
            m['last_seen_frame'] = frame_actual
            m['lost_time'] = 0.0
            m['just_coasted'] = False
            m['quarantine_until'] = 0.0
            next_id += 1

    # --------- Vida y limpieza: elimina tracks tras timeout ---------
    ids_a_borrar = []
    for pid, m in list(track_meta.items()):
        if pid == '_clock':
            continue
        if pid not in {p['id'] for p in personas_def}:
            # sigue coasting
            m['lost_time'] = m.get('lost_time', 0.0) + dt_seconds
        if m.get('lost_time', 0.0) > T_lost_seconds:
            ids_a_borrar.append(pid)
    for pid in ids_a_borrar:
        track_meta.pop(pid, None)
        kalmans.pop(pid, None)

    # --------- Desaparecidos para logging/salida ---------
    desaparecidos_def: Dict[int, np.ndarray] = {}
    for p in personas_anterior:
        if p['id'] not in {pp['id'] for pp in personas_def}:
            desaparecidos_def[p['id']] = p['keypoints']

    personas_anterior_nuevo = [dict(p) for p in personas_def]
    return personas_def, personas_anterior_nuevo, desaparecidos_def, next_id, kalmans, track_meta
