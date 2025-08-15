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

    poner_fila_a_ceros(output_csv, output_csv, [366,367, 368, 369, 400])