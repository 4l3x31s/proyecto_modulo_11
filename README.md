# Reconocedor de Billetes Bolivianos

Sistema de reconocimiento de billetes bolivianos utilizando PyTorch (GPU) y OCR.


## Participantes
- Alexeis Vladimir Carrillo Pinaya
- Fernando Luis Fernandez Castro

## Requisitos

### Python
- **Python 3.12.10** (recomendado)
- También funciona con Python 3.12.x

### Hardware
- GPU NVIDIA con soporte CUDA (RTX 3600 verificada)
- Cámara web

### Software Adicional
- **Tesseract OCR**: https://github.com/UB-Mannheim/tesseract/wiki
  - Instalar en: `C:\Program Files\Tesseract-OCR`
  - Agregar al PATH del sistema

## Instalación

### 1. Crear entorno virtual

```bash
# Desde la carpeta del proyecto
py -3.12 -m venv venv
```

### 2. Activar entorno virtual

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Instalar PyTorch con CUDA

```bash
# IMPORTANTE: Primero instalar PyTorch con soporte CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

## Estructura del Proyecto

```
proyecto/
├── dataset/              # Imágenes de entrenamiento
│   ├── 10/              # Billetes de 10 Bs
│   ├── 20/              # Billetes de 20 Bs
│   ├── 50/              # Billetes de 50 Bs
│   ├── 100/             # Billetes de 100 Bs
│   └── 200/             # Billetes de 200 Bs
├── train.py              # Script de entrenamiento
├── recognize.py          # Script de reconocimiento en tiempo real
├── billete_model.pth    # Modelo entrenado
└── requirements.txt     # Dependencias
```

## Uso

### Entrenamiento

```bash
py train.py
```

El entrenamiento genera el archivo `billete_model.pth`.

### Reconocimiento en Tiempo Real

```bash
py recognize.py
```

Presionar `q` para salir.

## Dependencias

| Paquete | Versión Mínima |
|---------|----------------|
| torch | 2.0.0 |
| torchvision | 0.15.0 |
| opencv-python | 4.8.0 |
| opencv-python-headless | 4.8.0 |
| numpy | 1.24.0 |
| Pillow | 10.0.0 |
| scikit-learn | 1.3.0 |
| tqdm | 4.65.0 |
| pytesseract | 0.3.10 |

## Características

- **Reconocimiento de corte**: 10, 20, 50, 100, 200 Bolivianos
- **Reconocimiento de serie**: A, B, C
- **Lectura de número de serie**: OCR (ej: 083594277 A)
- **GPU**: Utiliza CUDA si está disponible

## Solución de Problemas

### Error de Tesseract
Si aparece error de Tesseract, verificar que esté instalado en:
```
C:\Program Files\Tesseract-OCR\tesseract.exe
```

### GPU no detectada
El código automáticamente usa CPU si CUDA no está disponible. Verificar drivers de NVIDIA actualizados.

### Error al cargar modelo
Si hay error de "size mismatch", re-entrenar el modelo:
```bash
py train.py
```
