# tracking_system/demo_loop.py
import numpy as np
from service_client import predict_next_pose

smplx_to_openpose_xy = {
    0:  ("jaw_x",            "jaw_y"),            # Nose ≈ head
    1:  ("neck_x",            "neck_y"),            # Neck
    2:  ("right_shoulder_x",  "right_shoulder_y"),  # RShoulder
    3:  ("right_elbow_x",     "right_elbow_y"),     # RElbow
    4:  ("right_wrist_x",     "right_wrist_y"),     # RWrist
    5:  ("left_shoulder_x",   "left_shoulder_y"),   # LShoulder
    6:  ("left_elbow_x",      "left_elbow_y"),      # LElbow
    7:  ("left_wrist_x",      "left_wrist_y"),      # LWrist

    8:  ("pelvis_x",          "pelvis_y"),          # MidHip
    9:  ("right_hip_x",       "right_hip_y"),       # RHip
    10: ("right_knee_x",      "right_knee_y"),      # RKnee
    11: ("right_ankle_x",     "right_ankle_y"),     # RAnkle
    12: ("left_hip_x",        "left_hip_y"),        # LHip
    13: ("left_knee_x",       "left_knee_y"),       # LKnee
    14: ("left_ankle_x",      "left_ankle_y"),      # LAnkle

    15: ("right_eye_x", "right_eye_y"),  # REye    – no está en SMPL-X CSV
    16: ("left_eye_x", "left_eye_y"),  # LEye
    17: (),  # REar
    18: (),  # LEar

    19: ("left_foot_x",       "left_foot_y"),       # LBigToe ← left_foot
    20: (),  # LSmallToe
    21: (),  # LHeel

    22: ("right_foot_x",      "right_foot_y"),      # RBigToe ← right_foot
    23: (),  # RSmallToe
    24: (),  # RHeel
}


def limpiar_keypoints_por_mapeo(keypoints, smplx_to_openpose_xy=smplx_to_openpose_xy):
    # keypoints: np.ndarray de shape (frames, n_keypoints*2)
    # smplx_to_openpose_xy: dict con índices y tuplas de nombres o ()
    indices_a_mantener = []
    for idx, names in smplx_to_openpose_xy.items():
        if names:  # Si la tupla no está vacía
            indices_a_mantener.extend([idx*2, idx*2+1])  # x, y

    # Filtra las columnas
    keypoints_limpios = keypoints[indices_a_mantener]
    return keypoints_limpios