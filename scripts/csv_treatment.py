import csv
import os
from random import random

def trasladar_csv_x(input_csv, output_csv, dx=0.0):
    """
    Aplica una traslación dx a todas las coordenadas X del CSV,
    solo si el valor original no es 0 ni campo vacío.
    Asume que las coordenadas están en columnas: frame, joint_0_x, joint_0_y, joint_1_x, joint_1_y, ...
    """
    with open(input_csv, 'r') as fin, open(output_csv, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)  # Escribe la cabecera

        for fila in reader:
            new_row = [fila[0]]  # frame
            for i, val in enumerate(fila[1:]):
                # Determina si es coordenada X (índice par: 0,2,4,...)
                if i % 2 == 0:
                    if val.strip() != '' and float(val) != 0.0:
                        x_new = float(val) + dx
                        new_row.append(x_new)
                    else:
                        new_row.append(val)
                else:
                    new_row.append(val)
            writer.writerow(new_row)



def apilar_csvs_por_filas(input_csvs, output_csv):
    """
    Toma una lista de rutas de CSVs y crea un nuevo CSV donde:
    - Primero van todas las primeras filas de cada CSV,
    - Luego todas las segundas filas, etc.
    Si un CSV se queda sin filas, simplemente se ignora en las siguientes rondas.
    Asume que todos los CSVs tienen la misma cabecera.
    """
    # Abrimos todos los archivos y leemos sus filas (sin cabecera)
    filas_por_csv = []
    headers = []
    for path in input_csvs:
        with open(path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            headers.append(header)
            filas = list(reader)
            filas_por_csv.append(filas)
    # Comprobamos que todas las cabeceras son iguales
    if not all(h == headers[0] for h in headers):
        raise ValueError("Las cabeceras de los CSVs no coinciden.")

    # Encontrar el máximo número de filas entre todos los CSVs
    max_filas = max(len(filas) for filas in filas_por_csv)

    # Escribimos el nuevo CSV apilando por filas
    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(headers[0])
        for i in range(max_filas):
            for filas in filas_por_csv:
                if i < len(filas):
                    writer.writerow(filas[i])

def poner_fila_a_ceros(input_csv, output_csv, fila_idx=None):
    """
    Escribe un nuevo CSV igual al original pero con todos los valores (excepto el frame) de una fila (frame) puestos a 0.
    Si fila_idx es None, elige una fila aleatoria (sin contar la cabecera).
    """
    with open(input_csv, 'r') as fin:
        reader = list(csv.reader(fin))
        header = reader[0]
        filas = reader[1:]

    if not filas:
        raise ValueError("El CSV no contiene datos.")

    if fila_idx is None:
        fila_idx = random.randint(0, len(filas) - 1)

    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(header)
        for i, fila in enumerate(filas):
            if i in fila_idx:
                # Mantén el frame, pon el resto a 0
                new_row = [fila[0]] + ['0'] * (len(fila) - 1)
                writer.writerow(new_row)
            else:
                writer.writerow(fila)
    print(f"Fila {fila_idx} puesta a ceros en: {output_csv}")


def convertir_kp(csv_openpose, output_csv):
    """
    Convierte un CSV de OpenPose a formato apilado por persona:
    frame,joint_0_x,joint_0_y,...,joint_24_x,joint_24_y
    Una fila por persona y frame.
    """
    import csv
    from collections import defaultdict

    # Diccionario: {(frame, persona): [x0, y0, x1, y1, ..., x24, y24]}
    personas = defaultdict(lambda: [0.0]*50)

    with open(csv_openpose, 'r') as fin:
        reader = csv.reader(fin)
        header = next(reader)
        for fila in reader:
            frame = int(fila[1])
            persona = int(fila[2])
            joint = int(fila[3])
            x = float(fila[4])
            y = float(fila[5])
            # Coloca x, y en la posición correspondiente
            personas[(frame, persona)][joint*2] = x
            personas[(frame, persona)][joint*2+1] = y

    # Escribir el CSV apilado
    with open(output_csv, 'w', newline='') as fout:
        writer = csv.writer(fout)
        # Intercalar x, y
        header = ['frame'] + [f'joint_{i}_{c}' for i in range(25) for c in ('x','y')]
        writer.writerow(header)
        # Ordenar por frame y persona
        for (frame, persona) in sorted(personas.keys()):
            row = [frame] + personas[(frame, persona)]
            writer.writerow(row)




if __name__ == "__main__":
    input_csv = "../csv/A14_-_stand_to_skip_stageii_xz_openpose.csv"
    output_csv = "../csv/traslaciones/A14_-_stand_to_skip_stageii_xz_openpose_trasladado.csv"
    dx = 2.0  # Traslación en X
    # Crea el directorio de salida si no existe
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    #trasladar_csv_x(input_csv, output_csv, dx)
    print(f"Archivo trasladado guardado en: {output_csv}")
    
    input_csvs = [
        "../csv/A15_-_skip_to_stand_stageii_xz_openpose.csv",
        "../csv/A14_-_stand_to_skip_stageii_xz_openpose.csv",
        "../csv/A2_-_Sway_stageii_xz_openpose.csv",
        "../csv/A1-_Stand_stageii_xz_openpose.csv"
    ]
    output_csv = "../csv/apilado.csv"
    apilar_csvs_por_filas(input_csvs, output_csv)
    print(f"Archivo apilado guardado en: {output_csv}")

    csv_openpose = "../csv/output_keypoints_0.csv"
    convertir_kp(csv_openpose, "../csv/openpose_kp.csv")

    poner_fila_a_ceros(output_csv, output_csv, [366,367, 368, 369, 400])

'''
    input_csv = "../csv/openpose_kp.csv"
    output_csv = "../csv/openpose_kp_invertido.csv"

    # Leer todo el CSV y encontrar el máximo Y
    rows = []
    max_y = 0
    with open(input_csv, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            # Suponemos que las columnas son: frame, x0, y0, x1, y1, ..., xn, yn
            y_values = [float(row[i]) for i in range(2, len(row), 2) if row[i].strip() != ""]
            if y_values:
                max_y = max(max_y, max(y_values))
            rows.append(row)

    # Invertir las Y y guardar el nuevo CSV
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            new_row = row[:]
            for i in range(2, len(row), 2):
                if row[i].strip() != "":
                    y = float(row[i])
                    new_row[i] = str(max_y - y)
            writer.writerow(new_row)

    print(f"Archivo guardado como {output_csv}")

    input_csv = "../csv/openpose_kp_invertido.csv"
    output_csv = "../csv/openpose_kp_interpolado.csv"

    with open(input_csv, newline='') as f:
        reader = list(csv.reader(f))
        header = reader[0]
        data = reader[1:]

    new_rows = [header]
    frame_counter = 0

    for i in range(len(data) - 1):
        row1 = data[i]
        row2 = data[i + 1]
        # Escribimos la primera fila, actualizando el frame
        row1_new = row1[:]
        row1_new[0] = str(frame_counter)
        new_rows.append(row1_new)
        frame_counter += 1
        # Interpolamos (promedio) para cada columna numérica, frame entero siguiente
        interp = []
        interp.append(str(frame_counter))  # Frame entero siguiente
        for v1, v2 in zip(row1[1:], row2[1:]):
            try:
                interp.append(str((float(v1) + float(v2)) / 2))
            except ValueError:
                interp.append("")
        new_rows.append(interp)
        frame_counter += 1

    # Añadimos la última fila, con el siguiente frame entero
    row_last = data[-1][:]
    row_last[0] = str(frame_counter)
    new_rows.append(row_last)

    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerows(new_rows)

    print(f"Archivo guardado como {output_csv}")

'''