# ---------------------------
# Fernando Luis Fernandez Castro
# Alexeis Vladimir Carrillo Pinaya
# BITÁCORA:
# Prueba del modelo entrenado (MobileNetV2) con imágenes fijas en vez de cámara.
# Solo muestra la imagen con la predicción si la probabilidad supera el 80%.
# Redimensiona imágenes muy grandes para que se vean completas sin afectar el tamaño del texto.
# Además, anuncia la predicción en voz para cada imagen.
# ---------------------------

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import pyttsx3

# ---------------------------
# PARTE 1: CARGAR MODELO
# ---------------------------
model_path = "mejor_modelo_billetes.h5"
model = load_model(model_path)

# Cargar clases desde archivo generado en entrenamiento
if os.path.exists("class_indices.json"):
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    clases = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
else:
    clases = ['10', '20', '50', '100', '200']

IMG_SIZE = 224
test_dir = "dataset/test_images"

# ---------------------------
# PARTE 2: INICIALIZAR MOTOR DE VOZ
# ---------------------------
engine = pyttsx3.init()  # Inicializar solo una vez
engine.setProperty('rate', 150)  # Velocidad de la voz
engine.setProperty('volume', 1.0)  # Volumen máximo

# ---------------------------
# PARTE 3: RECORRER IMÁGENES DE PRUEBA
# ---------------------------
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    if not img_file.lower().endswith((".png", ".jpg", ".jpeg")):
        continue  # Saltar archivos no imagen

    # Cargar imagen original
    img = cv2.imread(img_path)
    if img is None:
        continue
    original_img = img.copy()  # Guardar copia para mostrar tamaño completo

    # Preprocesamiento para el modelo
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_norm, axis=0)  # Añadir batch

    # Predicción
    pred = model.predict(img_input, verbose=0)
    clase_pred = clases[np.argmax(pred)]
    prob = np.max(pred)

    # Mostrar solo si probabilidad > 0.65
    if prob > 0.65:
        texto = f"{clase_pred} Bs ({prob*100:.1f}%)"

        # Redimensionar imagen si es muy grande
        max_height = 600
        max_width = 800
        h, w = original_img.shape[:2] 
        scale = min(max_width / w, max_height / h, 1.0)
        display_img = cv2.resize(original_img, (int(w * scale), int(h * scale)))

        # Texto fijo
        cv2.putText(display_img, texto, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Mostrar ventana
        cv2.imshow(f"Predicción - {img_file}", display_img)
        print(f"[OK] Imagen: {img_file} -> {texto}")

        # Leer en voz la predicción
        #engine = pyttsx3.init()  # Inicializar solo una vez
        #engine.setProperty('rate', 150)  # Velocidad de la voz
        #engine.setProperty('volume', 1.0)  # Volumen máximo              
        engine.say(f"Billete de {clase_pred} Bolivianos")
        engine.say(f"Billete de {clase_pred} Bolivianos")
        engine.runAndWait()

        # Esperar a que se cierre la ventana o presione cualquier tecla
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"[INFO] Imagen: {img_file} -> Probabilidad baja ({prob*100:.1f}%), no se muestra predicción")

print("🔚 Test completado.")