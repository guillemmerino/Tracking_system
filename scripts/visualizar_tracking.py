from io import BytesIO
import matplotlib.pyplot as plt
import imageio
import numpy as np
import os

def dibujar_personas_en_frame(personas, ax, color_map):
    skeleton = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
        (0, 15), (0, 16), (14, 19), (11, 22),
    ]
    for persona in personas:
        keypoints = persona['keypoints'].reshape(-1, 2)
        id_ = persona['id']
        color = color_map[id_ % len(color_map)]
        xs, ys = keypoints[:, 0], keypoints[:, 1]
        ax.scatter(xs, ys, color=color, label=f'ID {id_}')
        for i, (x, y) in enumerate(zip(xs, ys)):
            ax.text(x, y, str(i), fontsize=6, color=color)
        for a, b in skeleton:
            if a < len(xs) and b < len(xs):
                ax.plot([xs[a], xs[b]], [ys[a], ys[b]], color=color, linewidth=2)
        ax.text(xs[0], ys[0], f'ID {id_}', fontsize=10, color=color, weight='bold')

def visualizar_tracking(frames_personas, output_gif="tracking.gif"):
    color_map = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta', 'lime', 'gold']
    images = []
    fig_width, fig_height, dpi = 6, 6, 100  # Tamaño fijo
    for frame_idx, personas in enumerate(frames_personas):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        dibujar_personas_en_frame(personas, ax, color_map)
        ax.set_title(f"Frame {frame_idx}")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        img = imageio.imread(buf)
        images.append(img)
        buf.close()
    imageio.mimsave(output_gif, images, duration=0.2)
    print(f"GIF guardado en: {output_gif}")