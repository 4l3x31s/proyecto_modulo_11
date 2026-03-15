# ---------------------------
# Fernando Luis Fernandez Castro
# Alexeis Vladimir Carrillo Pinaya
# BITÁCORA:
# Segmentación y recorte de billetes usando archivos JSON generados por LabelMe.
# Se recorre cada carpeta de denominación dentro de dataset_original.
# ---------------------------

import os
import json
import cv2
import numpy as np

# Carpeta raíz con imágenes y JSON
input_root = "dataset_original"
# Carpeta donde se guardarán las imágenes recortadas
output_root = "dataset/train"
os.makedirs(output_root, exist_ok=True)

# Iterar sobre cada subcarpeta (denominación)
for label in os.listdir(input_root):
    label_dir = os.path.join(input_root, label)
    if not os.path.isdir(label_dir):
        continue

    save_dir = os.path.join(output_root, label)
    os.makedirs(save_dir, exist_ok=True)

    # Iterar sobre cada archivo en la carpeta de la denominación
    for file in os.listdir(label_dir):
        if file.endswith(".json"):
            json_path = os.path.join(label_dir, file)
            with open(json_path, "r") as f:
                data = json.load(f)

            img_name = data["imagePath"]
            img_path = os.path.join(label_dir, img_name)
            img = cv2.imread(img_path)

            if img is None:
                print(f"[ERROR] No se pudo leer la imagen {img_path}")
                continue

            # Iterar sobre los polígonos (puede haber más de uno)
            for shape in data["shapes"]:
                points = np.array(shape["points"], dtype=np.int32)

                # Crear máscara
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)

                # Aplicar máscara
                segmented = cv2.bitwise_and(img, img, mask=mask)

                # Recortar usando boundingRect
                x, y, w, h = cv2.boundingRect(points)
                cropped = segmented[y:y+h, x:x+w]

                # Redimensionar a 224x224
                resized = cv2.resize(cropped, (224, 224))

                # Guardar imagen recortada
                base_name = os.path.splitext(img_name)[0]
                save_path = os.path.join(save_dir, f"{base_name}_recortada.png")
                cv2.imwrite(save_path, resized)

                print(f"[OK] Imagen guardada: {save_path}")

print("Segmentación y recorte completados.")