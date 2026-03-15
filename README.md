# Sistema de Reconocimiento de Billetes - Proyecto de Maestría

**Autores:** 

        Fernando Luis Fernandez Castro 
        Alexeis Vladimir Carrillo Pinaya

Sistema de reconocimiento de billetes Bolivianos utilizando Transfer Learning con MobileNetV2 y TensorFlow.

## Descripción

Este proyecto implementa un modelo de deep learning para clasificar billetes Bolivianos (20 y 50 Bs, según los datos actuales). Utiliza:

- **MobileNetV2** preentrenado en ImageNet como base
- **Transfer Learning** con fine-tuning de las últimas capas
- **Aumento de datos** para robustez del modelo

## Estructura del Proyecto

```
ProyectoMaestria/
├── 0_SegmentarYRecortar.py   # Script para segmentar/recortar imágenes usando anotaciones de LabelMe
├── 1_MobileNetV2Train.py     # Script de entrenamiento del modelo
├── 2_testModeloImagenesTest.py # Script de prueba con imágenes
├── class_indices.json         # Mapeo de clases (géneros del modelo)
├── requierements.txt          # Dependencias del proyecto
├── modelo_billetes_mobilenetv2_final.h5  # Modelo final entrenado
├── mejor_modelo_billetes.h5   # Mejor modelo guardado durante entrenamiento
└── dataset/
    ├── train/                 # Imágenes de entrenamiento
    │   ├── 20/                # Billetes de 20 Bs
    │   └── 50/                # Billetes de 50 Bs
    ├── val/                   # Imágenes de validación
    │   ├── 20/
    │   └── 50/
    └── test_images/           # Imágenes para pruebas
```

## Requisitos

- Python 3.10
- TensorFlow 2.15.0
- OpenCV
- NumPy 1.26.4
- scikit-learn
- pyttsx3 (para síntesis de voz)

### Instalación

```bash
# Crear entorno virtual
py -3.10 -m venv env

# Activar entorno (Windows)
env\Scripts\activate

# Instalar dependencias
pip install --upgrade pip
pip install tensorflow==2.15.0
pip install numpy==1.26.4 opencv-python==4.9.0.80
pip install pillow scipy scikit-learn pyttsx3
```

## Scripts

### 1. Segmentación y Recorte (`0_SegmentarYRecortar.py`)

Procesa imágenes originales con anotaciones JSON de LabelMe para:
- Segmentar la región del billete usando polígonos
- Recortar y redimensionar a 224x224
- Guardar en la estructura de carpetas por clase

**Uso:** Ejecutar desde la carpeta raíz con imágenes en `dataset_original/`

### 2. Entrenamiento (`1_MobileNetV2Train.py`)

Entrena el modelo con:
- **Entradas:** Imágenes de entrenamiento en `dataset/train/`
- **Validación:** Imágenes en `dataset/val/`
- **Aumento de datos:** Rotación, zoom, brillo, flip horizontal
- **Fine-tuning:** Descongela últimas 30 capas de MobileNetV2
- **Callbacks:** EarlyStopping y ModelCheckpoint

**Parámetros:**
- Tamaño de imagen: 224x224
- Batch size: 16
- Epochs: 50 (con early stopping)
- Learning rate: 1e-5 (fine-tuning)

**Salida:**
- `mejor_modelo_billetes.h5` - Mejor modelo según validación
- `modelo_billetes_mobilenetv2_final.h5` - Modelo final
- `class_indices.json` - Índices de clases

### 3. Predicción (`2_testModeloImagenesTest.py`)

Prueba el modelo con imágenes en `dataset/test_images/`:
- Muestra ventana con imagen y predicción
- Annuncia el resultado en voz (pyttsx3)
- Solo muestra predicciones con confianza > 65%

## Clases Actuales

Según `class_indices.json`:
- `0`: 20 Bs
- `1`: 50 Bs

## Métricas del Modelo

El script de entrenamiento genera:
- Matriz de confusión
- F1-score ponderado

## Notas

- El modelo actual está entrenado solo con clases 20 y 50 Bs
- Para agregar más denominaciones (10, 100, 200), agregar carpetas en `dataset/train/` y `dataset/val/`
- Las imágenes deben estar redimensionadas a 224x224 para inferencia
