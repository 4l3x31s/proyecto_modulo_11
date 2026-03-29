import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

BILL_CLASSES = ['10', '20', '50', '100', '200']
SERIES = ['A', 'B', 'C']
IMG_SIZE = 224


class BilleteCNN(torch.nn.Module):
    def __init__(self, num_classes=5, num_series=3):
        super(BilleteCNN, self).__init__()
        
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
            
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2, 2),
        )
        
        self.shared = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.5),
        )
        
        self.fc_class = torch.nn.Linear(256, num_classes)
        self.fc_serie = torch.nn.Linear(256, num_series)
    
    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        
        class_out = self.fc_class(x)
        serie_out = self.fc_serie(x)
        
        return class_out, serie_out


def load_model(model_path='billete_model.pth'):
    print("=" * 60)
    print("  RECONOCEDOR DE BILLETES BOLIVIANOS")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n  [GPU] CUDA detectada!")
        print(f"  [GPU] Tarjeta: {torch.cuda.get_device_name(0)}")
        print(f"  [GPU] Usando GPU para inferencia\n")
    else:
        device = torch.device('cpu')
        print(f"\n  [CPU] CUDA no disponible")
        print(f"  [CPU] Usando CPU para inferencia\n")
    
    print("-" * 60)
    
    if not os.path.exists(model_path):
        print(f"\nError: No se encontró el modelo '{model_path}'")
        print("Primero debes entrenar el modelo con: python train.py")
        return None, None, None
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model = BilleteCNN(num_classes=len(BILL_CLASSES), num_series=len(SERIES)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    classes = checkpoint.get('classes', BILL_CLASSES)
    series = checkpoint.get('series', SERIES)
    print(f"Modelo cargado.")
    print(f"Clases de billete: {classes}")
    print(f"Series: {series}")
    
    return model, device, classes


def preprocess_frame(frame):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(pil_image).unsqueeze(0)
    return tensor


def detect_billete(model, frame, device):
    with torch.no_grad():
        input_tensor = preprocess_frame(frame).to(device)
        class_out, serie_out = model(input_tensor)
        
        class_probs = torch.nn.functional.softmax(class_out, dim=1)
        serie_probs = torch.nn.functional.softmax(serie_out, dim=1)
        
        confidence_class, predicted_class_idx = torch.max(class_probs, 1)
        confidence_serie, predicted_serie_idx = torch.max(serie_probs, 1)
        
        predicted_class = BILL_CLASSES[predicted_class_idx.item()]
        predicted_serie = SERIES[predicted_serie_idx.item()]
        
        class_prob_percent = confidence_class.item() * 100
        serie_prob_percent = confidence_serie.item() * 100
        
        all_class_probs = class_probs[0].cpu().numpy()
        all_serie_probs = serie_probs[0].cpu().numpy()
    
    return predicted_class, predicted_serie, class_prob_percent, serie_prob_percent, all_class_probs, all_serie_probs


def extract_serial_and_letter(frame):
    try:
        h, w = frame.shape[:2]
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        regions = [
            gray[int(h*0.65):int(h*0.9), int(w*0.1):int(w*0.5)],
            gray[int(h*0.65):int(h*0.9), int(w*0.5):int(w*0.9)],
            gray[int(h*0.65):int(h*0.9), :],
        ]
        
        for i, roi in enumerate(regions):
            roi = cv2.resize(roi, (roi.shape[1]*2, roi.shape[0]*2))
            
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(roi, config=custom_config)
            
            numbers = ''.join(c for c in text if c.isdigit())
            letters = ''.join(c for c in text if c.isalpha()).upper()
            
            letter = None
            for l in letters:
                if l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                    letter = l
                    break
            
            if numbers and len(numbers) >= 6:
                return numbers, letter
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 8))
        dilate = cv2.dilate(binary, kernel, iterations=2)
        
        contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if 50 < cw < 300 and 15 < ch < 60:
                roi = binary[y:y+ch, x:x+cw]
                roi = cv2.resize(roi, (roi.shape[1]*3, roi.shape[0]*3))
                
                text = pytesseract.image_to_string(roi, config=r'--oem 3 --psm 7')
                numbers = ''.join(c for c in text if c.isdigit())
                letters = ''.join(c for c in text if c.isalpha()).upper()
                
                letter = None
                for l in letters:
                    if l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                        letter = l
                        break
                
                if numbers and len(numbers) >= 6:
                    return numbers, letter
        
        return None, None
    except Exception as e:
        print(f"OCR Error: {e}")
        return None, None


def recognize_from_camera():
    model, device, classes = load_model()
    
    if model is None:
        return
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("\nError: No se pudo abrir la cámara web")
        print("Asegúrate de que la cámara no esté siendo usada por otra aplicación.")
        return
    
    print("\n" + "=" * 50)
    print("CÁMARA INICIADA")
    print("Presiona 'q' para salir")
    print("=" * 50 + "\n")
    
    window_name = 'Reconocimiento de Billetes Bolivianos'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)
    
    last_serial = ""
    last_serie_letter = ""
    ocr_attempts = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error al capturar frame")
            break
        
        prediction, serie_ia, conf_class, conf_serie, class_probs, serie_probs = detect_billete(model, frame, device)
        
        ocr_attempts += 1
        if ocr_attempts % 5 == 0:
            serial_num, serie_letter = extract_serial_and_letter(frame)
            ocr_attempts = 0
            if serial_num and len(serial_num) >= 4:
                last_serial = serial_num
            if serie_letter:
                last_serie_letter = serie_letter
        
        color = (0, 255, 0) if conf_class > 70 else (0, 165, 255)
        
        h, w = frame.shape[:2]
        panel_width = 420
        panel_height = 240
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.putText(frame, f"BILLETE: {prediction} Bolivianos", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        serie_display = last_serie_letter if last_serie_letter else serie_ia
        cv2.putText(frame, f"SERIE: {serie_display}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 200, 100), 2)
        
        if last_serial:
            full_serial = f"{last_serial} {last_serie_letter}" if last_serie_letter else last_serial
            cv2.putText(frame, f"No. SERIE: {full_serial}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)
        else:
            cv2.putText(frame, "No. SERIE: Buscando...", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 1)
        
        cv2.putText(frame, f"Confianza clase: {conf_class:.1f}%", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        y_offset = 165
        cv2.putText(frame, "Billetes:", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        for i, (cls, prob) in enumerate(zip(BILL_CLASSES, class_probs)):
            prob_text = f"{cls}: {prob*100:.1f}%"
            color_prob = (0, 255, 0) if prob > 0.5 else (200, 200, 200)
            cv2.putText(frame, prob_text, (110, y_offset + i*18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_prob, 1)
        
        cv2.imshow(window_name, frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nCámara cerrada.")


if __name__ == '__main__':
    recognize_from_camera()
