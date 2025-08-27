import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
from matplotlib.patches import Circle

def get_project_root():
    """Obtener la raíz del proyecto"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)

PROJECT_ROOT = get_project_root()
OPENPOSE_DIR = os.path.join(PROJECT_ROOT, "csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_videos")

# Conexiones OpenPose para dibujar el esqueleto
OPENPOSE_CONNECTIONS = [
    # Cabeza y cuello
    (0, 1),   # Nose -> Neck
    (1, 15),  # Neck -> REye  
    (1, 16),  # Neck -> LEye
    (15, 17), # REye -> REar
    (16, 18), # LEye -> LEar
    
    # Brazo derecho
    (1, 2),   # Neck -> RShoulder
    (2, 3),   # RShoulder -> RElbow
    (3, 4),   # RElbow -> RWrist
    
    # Brazo izquierdo
    (1, 5),   # Neck -> LShoulder
    (5, 6),   # LShoulder -> LElbow
    (6, 7),   # LElbow -> LWrist
    
    # Torso
    (1, 8),   # Neck -> MidHip
    (8, 9),   # MidHip -> RHip
    (8, 12),  # MidHip -> LHip
    
    # Pierna derecha
    (9, 10),  # RHip -> RKnee
    (10, 11), # RKnee -> RAnkle
    (11, 22), # RAnkle -> RBigToe
    (22, 23), # RBigToe -> RSmallToe
    (11, 24), # RAnkle -> RHeel
    
    # Pierna izquierda  
    (12, 13), # LHip -> LKnee
    (13, 14), # LKnee -> LAnkle
    (14, 19), # LAnkle -> LBigToe
    (19, 20), # LBigToe -> LSmallToe
    (14, 21), # LAnkle -> LHeel
]

# Colores para diferentes partes del cuerpo
BODY_COLORS = {
    'head': '#FF6B6B',      # Rojo
    'torso': '#4ECDC4',     # Verde azulado
    'right_arm': '#45B7D1',  # Azul
    'left_arm': '#96CEB4',   # Verde claro  
    'right_leg': '#FFEAA7',  # Amarillo
    'left_leg': '#DDA0DD',   # Violeta
    'default': '#95A5A6'     # Gris
}

def load_openpose_csv(csv_path):
    """Cargar datos OpenPose desde CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"El archivo {csv_path} no existe")
    
    print(f"Cargando datos OpenPose desde: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Extraer coordenadas de articulaciones
    n_frames = len(df)
    n_joints = 25
    
    openpose_data = np.zeros((n_frames, n_joints, 2))
    
    for joint_idx in range(n_joints):
        x_col = f'joint_{joint_idx}_x'
        y_col = f'joint_{joint_idx}_y'
        
        if x_col in df.columns and y_col in df.columns:
            openpose_data[:, joint_idx, 0] = df[x_col].values
            openpose_data[:, joint_idx, 1] = df[y_col].values
        else:
            openpose_data[:, joint_idx, :] = np.nan
    
    print(f"Datos cargados: {n_frames} frames, {n_joints} articulaciones")
    
    # Contar articulaciones válidas
    valid_joints = np.sum(~np.isnan(openpose_data[0, :, 0]))
    print(f"Articulaciones válidas: {valid_joints}/25")
    
    return openpose_data

def get_connection_color(connection):
    """Obtener color para una conexión específica"""
    joint1, joint2 = connection
    
    # Cabeza/cuello
    if joint1 in [0, 1, 15, 16, 17, 18] or joint2 in [0, 1, 15, 16, 17, 18]:
        return BODY_COLORS['head']
    # Brazo derecho
    elif joint1 in [2, 3, 4] or joint2 in [2, 3, 4]:
        return BODY_COLORS['right_arm']
    # Brazo izquierdo  
    elif joint1 in [5, 6, 7] or joint2 in [5, 6, 7]:
        return BODY_COLORS['left_arm']
    # Pierna derecha
    elif joint1 in [9, 10, 11, 22, 23, 24] or joint2 in [9, 10, 11, 22, 23, 24]:
        return BODY_COLORS['right_leg']
    # Pierna izquierda
    elif joint1 in [12, 13, 14, 19, 20, 21] or joint2 in [12, 13, 14, 19, 20, 21]:
        return BODY_COLORS['left_leg']
    # Torso
    else:
        return BODY_COLORS['torso']

def create_openpose_animation(openpose_data, output_path, fps=30, interval=50):
    """Crear animación de la pose OpenPose"""
    n_frames, n_joints, _ = openpose_data.shape
    
    # Configurar figura
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Calcular límites del plot
    valid_data = openpose_data[~np.isnan(openpose_data)]
    if len(valid_data) > 0:
        margin = 0.2
        x_min, x_max = np.nanmin(openpose_data[:, :, 0]), np.nanmax(openpose_data[:, :, 0])
        y_min, y_max = np.nanmin(openpose_data[:, :, 1]), np.nanmax(openpose_data[:, :, 1])
        
        x_range = x_max - x_min
        y_range = y_max - y_min
        
        ax.set_xlim(x_min - margin * x_range, x_max + margin * x_range)
        ax.set_ylim(y_min - margin * y_range, y_max + margin * y_range)
    else:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
    
    ax.set_aspect('equal')
    #ax.invert_yaxis()  # OpenPose tiene Y invertida
    ax.set_title('Pose OpenPose Reconstruction', color='white', fontsize=16, pad=20)
    ax.grid(True, alpha=0.3, color='gray')
    
    # Remover ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Elementos de la animación
    lines = []  # Para conexiones
    points = []  # Para articulaciones
    
    # Crear líneas para conexiones
    for connection in OPENPOSE_CONNECTIONS:
        line, = ax.plot([], [], linewidth=3, alpha=0.8, 
                       color=get_connection_color(connection))
        lines.append((line, connection))
    
    # Crear puntos para articulaciones
    for joint_idx in range(n_joints):
        point = Circle((0, 0), radius=0.03, alpha=0.9, 
                      color='white', ec='black', linewidth=1)
        ax.add_patch(point)
        points.append(point)
    
    # Texto para información del frame
    frame_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                        fontsize=12, color='white', 
                        verticalalignment='top', fontweight='bold')
    
    def animate(frame):
        """Función de animación para cada frame"""
        current_pose = openpose_data[frame]
        
        # Actualizar conexiones
        for line, (joint1, joint2) in lines:
            x1, y1 = current_pose[joint1]
            x2, y2 = current_pose[joint2]
            
            # Solo dibujar si ambos puntos son válidos
            if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                line.set_data([x1, x2], [y1, y2])
                line.set_alpha(0.8)
            else:
                line.set_data([], [])
                line.set_alpha(0.0)
        
        # Actualizar puntos de articulaciones
        for joint_idx, point in enumerate(points):
            x, y = current_pose[joint_idx]
            if not (np.isnan(x) or np.isnan(y)):
                point.center = (x, y)
                point.set_alpha(0.9)
            else:
                point.set_alpha(0.0)
        
        # Actualizar texto del frame
        frame_text.set_text(f'Frame: {frame+1}/{n_frames}')
        
        return [line for line, _ in lines] + points + [frame_text]
    
    # Crear animación
    print(f"Creando animación con {n_frames} frames...")
    anim = animation.FuncAnimation(
        fig, animate, frames=n_frames, interval=interval, 
        blit=False, repeat=True
    )
    
    # Guardar como GIF
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Guardando animación en: {output_path}")
    
    anim.save(output_path, writer='pillow', fps=fps, 
              savefig_kwargs={'facecolor': 'black', 'edgecolor': 'none'})
    
    plt.close()
    return output_path

def create_comparison_plots(openpose_data, output_dir, n_samples=6):
    """Crear plots de comparación de diferentes frames"""
    n_frames = openpose_data.shape[0]
    sample_frames = np.linspace(0, n_frames-1, n_samples, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()
    
    for idx, frame in enumerate(sample_frames):
        ax = axes[idx]
        current_pose = openpose_data[frame]
        
        # Dibujar conexiones
        for connection in OPENPOSE_CONNECTIONS:
            joint1, joint2 = connection
            x1, y1 = current_pose[joint1]
            x2, y2 = current_pose[joint2]
            
            if not (np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2)):
                ax.plot([x1, x2], [y1, y2], linewidth=2, 
                       color=get_connection_color(connection), alpha=0.7)
        
        # Dibujar puntos
        for joint_idx in range(25):
            x, y = current_pose[joint_idx]
            if not (np.isnan(x) or np.isnan(y)):
                ax.scatter(x, y, s=50, c='red', zorder=5, alpha=0.8)
        
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.set_title(f'Frame {frame+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'comparison_frames.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Comparación de frames guardada: {comparison_path}")
    return comparison_path

def analyze_openpose_data(openpose_data):
    """Analizar calidad de los datos OpenPose"""
    n_frames, n_joints, _ = openpose_data.shape
    
    print(f"\n📊 ANÁLISIS DE DATOS OPENPOSE")
    print(f"{'='*40}")
    print(f"Frames totales: {n_frames}")
    print(f"Articulaciones: {n_joints}")
    
    # Calcular estadísticas de datos válidos
    valid_data = ~np.isnan(openpose_data)
    valid_joints_per_frame = np.sum(valid_data[:, :, 0], axis=1)
    valid_frames_per_joint = np.sum(valid_data[:, :, 0], axis=0)
    
    print(f"\nEstadísticas de completitud:")
    print(f"- Promedio de articulaciones válidas por frame: {valid_joints_per_frame.mean():.1f}/25")
    print(f"- Frames con todas las articulaciones: {np.sum(valid_joints_per_frame == 25)}")
    print(f"- Frames con >20 articulaciones: {np.sum(valid_joints_per_frame > 20)}")
    print(f"- Frames con <10 articulaciones: {np.sum(valid_joints_per_frame < 10)}")
    
    # Articulaciones más problemáticas
    joint_completeness = valid_frames_per_joint / n_frames * 100
    problematic_joints = np.where(joint_completeness < 80)[0]
    
    if len(problematic_joints) > 0:
        print(f"\nArticulaciones problemáticas (<80% completitud):")
        for joint_idx in problematic_joints:
            print(f"- Joint {joint_idx}: {joint_completeness[joint_idx]:.1f}%")
    
    return {
        'n_frames': n_frames,
        'n_joints': n_joints,
        'avg_valid_joints': valid_joints_per_frame.mean(),
        'complete_frames': np.sum(valid_joints_per_frame == 25),
        'joint_completeness': joint_completeness
    }

def process_openpose_file(csv_path, output_dir=None, create_gif=True, fps=15):
    """Procesar archivo OpenPose completo"""
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    try:
        # Cargar datos
        openpose_data = load_openpose_csv(csv_path)
        
        # Analizar calidad de datos
        stats = analyze_openpose_data(openpose_data)
        
        # Crear directorio de salida
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        file_output_dir = os.path.join(output_dir, base_name)
        os.makedirs(file_output_dir, exist_ok=True)
        
        # Crear comparación de frames
        create_comparison_plots(openpose_data, file_output_dir)
        
        if create_gif and stats['avg_valid_joints'] > 10:
            # Crear animación GIF
            gif_path = os.path.join(file_output_dir, f'{base_name}_openpose.gif')
            create_openpose_animation(openpose_data, gif_path, fps=fps)
            print(f"✓ GIF creado: {gif_path}")
            
            return gif_path, stats
        else:
            print("⚠️ Datos insuficientes para crear GIF de calidad")
            return None, stats
            
    except Exception as e:
        print(f"❌ Error procesando {csv_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    """Función principal"""
    if len(sys.argv) > 1:
        # Archivo específico
        csv_path = "/workspaces/Tracking_system/csv/openpose_kp_invertido.csv"
        if not os.path.isabs(csv_path):
            csv_path = os.path.join(PROJECT_ROOT, csv_path)
    else:
        # Buscar archivos OpenPose disponibles
        if os.path.exists(OPENPOSE_DIR):
            openpose_files = [f for f in os.listdir(OPENPOSE_DIR) if f.endswith('.csv')]
            
            if not openpose_files:
                print(f"❌ No se encontraron archivos OpenPose en {OPENPOSE_DIR}")
                print("💡 Ejecuta primero 'python3 secuencias_lstm.py' para generar datos OpenPose")
                return
            
            print("Archivos OpenPose disponibles:")
            for i, file in enumerate(openpose_files):
                print(f"  {i+1}. {file}")
            
            try:
                choice = int(input("\nSelecciona un archivo (número): ")) - 1
                if 0 <= choice < len(openpose_files):
                    csv_path = os.path.join(OPENPOSE_DIR, openpose_files[choice])
                else:
                    print("Selección inválida")
                    return
            except (ValueError, KeyboardInterrupt):
                print("Selección cancelada")
                return
        else:
            print(f"❌ Directorio OpenPose no encontrado: {OPENPOSE_DIR}")
            return
    
    if os.path.exists(csv_path):
        print(f"Procesando archivo OpenPose: {os.path.basename(csv_path)}")
        gif_path, stats = process_openpose_file(csv_path, fps=20)
        
        if gif_path:
            print(f"\n✅ ÉXITO!")
            print(f"📁 GIF creado: {gif_path}")
            print(f"📊 Estadísticas: {stats['complete_frames']} frames completos de {stats['n_frames']}")
            
            # Abrir en navegador si está disponible
            if 'BROWSER' in os.environ:
                os.system(f'"{os.environ["BROWSER"]}" "{gif_path}"')
        else:
            print(f"\n⚠️ No se pudo crear el GIF")
    else:
        print(f"❌ El archivo {csv_path} no existe")

if __name__ == "__main__":
    main()