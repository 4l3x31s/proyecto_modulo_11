# ---------------------------
# Fernando Luis Fernandez Castro
# Alexeis Vladimir Carrillo Pinaya
# BITÁCORA:
# Entrenamiento mejorado del modelo MobileNetV2 usando imágenes de billetes a color (10, 20, 50, 100, 200).
# Se cargan las imágenes de entrenamiento y validación, se aplica aumento de datos para robustez,
# se define MobileNetV2 preentrenado, se descongelan últimas capas para fine-tuning y se entrena.
# Se guardará el mejor modelo automáticamente y un diccionario con los índices de clase.
# ---------------------------

# type: ignore
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------------------
# PARTE 1: PREPARAR LOS DATOS
# ---------------------------

IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset/train"  # Carpeta con imágenes de entrenamiento organizadas por clase
val_dir = "dataset/val"      # Carpeta con imágenes de validación

# Generador para entrenamiento con aumento de datos
#Este bloque crea nuevas versiones modificadas de tus imágenes mientras entrenas (sin guardar nuevos archivos).
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Generador para validación (solo normalización)
val_datagen = ImageDataGenerator(rescale=1.0/255)

# Cargar imágenes desde carpetas
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"  # Ahora se trabaja con imágenes a color
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    color_mode="rgb"
)

# Guardar el mapeo de clases
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
print("Diccionario de clases guardado como class_indices.json")

# ---------------------------
# PARTE 2: CREAR EL MODELO
# ---------------------------

# Entrada para imágenes a color
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

# Cargar modelo base preentrenado (sin la parte superior)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_tensor=inputs)

# Congelar inicialmente todas las capas
base_model.trainable = False

# Añadir capas personalizadas para clasificación de billetes
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(train_generator.num_classes, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

# Compilar modelo inicial
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ---------------------------
# PARTE 3: FINE-TUNING
# ---------------------------

# Descongelar últimas capas de MobileNetV2 para ajuste fino
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Recompilar con menor learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---------------------------
# PARTE 4: CALLBACKS
# ---------------------------

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint("mejor_modelo_billetes.h5", monitor="val_accuracy", save_best_only=True)
]

# ---------------------------
# PARTE 5: ENTRENAR EL MODELO
# ---------------------------

EPOCHS = 50

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ---------------------------
# PARTE 6: GUARDAR EL MODELO FINAL
# ---------------------------

model.save("modelo_billetes_mobilenetv2_final.h5")
print("Modelo final guardado como modelo_billetes_mobilenetv2_final.h5")

# ---------------------------
# EVALUACIONES
# ---------------------------

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
import matplotlib.pyplot as plt

# Generar predicciones sobre el conjunto de validación
y_true = []
y_pred = []

for i in range(len(val_generator)):
    X_val, y_val = val_generator[i]
    preds = model.predict(X_val, verbose=0)
    y_true.extend(np.argmax(y_val, axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(train_generator.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Matriz de Confusión")
plt.show()

# F1-score
f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1-score ponderado: {f1:.4f}")