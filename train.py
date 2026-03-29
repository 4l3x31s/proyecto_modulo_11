import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
import random

BILL_CLASSES = ['10', '20', '50', '100', '200']
SERIES = ['A', 'B', 'C']
IMG_SIZE = 224


class BilleteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        self.series = []
        
        for class_idx, class_name in enumerate(BILL_CLASSES):
            class_path = self.root_dir / class_name
            if not class_path.exists():
                print(f"Advertencia: No se encontró la carpeta {class_name}")
                continue
            
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            for img_path in class_path.iterdir():
                if img_path.suffix.lower() in valid_extensions:
                    self.images.append(img_path)
                    self.labels.append(class_idx)
                    serie = self._extract_serie(img_path.name)
                    self.series.append(serie)
        
        if len(self.images) == 0:
            print("Error: No se encontraron imágenes en el dataset.")
            print(f"Carpetas esperadas: {BILL_CLASSES}")
            print(f"Ruta buscada: {self.root_dir}")
            raise ValueError("Dataset vacío - no hay imágenes para entrenar")
    
    def _extract_serie(self, filename):
        fname = filename.lower()
        if 'seriea' in fname or fname.startswith('a') or '_a' in fname:
            return 0
        elif 'serieb' in fname or fname.startswith('b') or '_b' in fname:
            return 1
        elif 'seriec' in fname or fname.startswith('c') or '_c' in fname:
            return 2
        return random.randint(0, 2)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error cargando {img_path}: {e}")
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        
        label = self.labels[idx]
        serie = self.series[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, serie


class BilleteCNN(nn.Module):
    def __init__(self, num_classes=5, num_series=3):
        super(BilleteCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.shared = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )
        
        self.fc_class = nn.Linear(256, num_classes)
        self.fc_serie = nn.Linear(256, num_series)
    
    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        
        class_out = self.fc_class(x)
        serie_out = self.fc_serie(x)
        
        return class_out, serie_out


def train_model(dataset_path, epochs=30, batch_size=8, learning_rate=0.001):
    print("=" * 60)
    print("  ENTRENAMIENTO DE RECONOCEDOR DE BILLETES")
    print("=" * 60)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\n  [GPU] CUDA detectada!")
        print(f"  [GPU] Tarjeta: {torch.cuda.get_device_name(0)}")
        print(f"  [GPU] CUDA Version: {torch.version.cuda}")
        print(f"  [GPU] Usando GPU para entrenamiento\n")
    else:
        device = torch.device('cpu')
        print(f"\n  [CPU] CUDA no disponible")
        print(f"  [CPU] Usando CPU para entrenamiento\n")
    
    print("-" * 60)
    
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print(f"\nBuscando dataset en: {dataset_path}")
    for class_name in BILL_CLASSES:
        class_path = Path(dataset_path) / class_name
        if class_path.exists():
            files = list(class_path.glob('*'))
            files = [f for f in files if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
            print(f"  - {class_name} Bolivianos: {len(files)} imágenes")
        else:
            print(f"  - {class_name} Bolivianos: No encontrada")
    
    try:
        dataset = BilleteDataset(dataset_path, transform=transform)
    except ValueError:
        return None
    
    print(f"\nTotal de imágenes: {len(dataset)}")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {len(train_dataset)}, Validation: {len(val_dataset)}")
    print(f"\nÉpocas: {epochs} | Batch: {batch_size} | LR: {learning_rate}")
    
    model = BilleteCNN(num_classes=len(BILL_CLASSES), num_series=len(SERIES)).to(device)
    
    criterion_class = nn.CrossEntropyLoss()
    criterion_serie = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_val_acc = 0.0
    
    print("\n" + "=" * 50)
    print("INICIANDO ENTRENAMIENTO...")
    print("=" * 50 + "\n")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_class_correct = 0
        train_serie_correct = 0
        train_total = 0
        
        for images, labels, series in train_loader:
            images, labels, series = images.to(device), labels.to(device), series.to(device)
            
            optimizer.zero_grad()
            
            class_out, serie_out = model(images)
            
            loss_class = criterion_class(class_out, labels)
            loss_serie = criterion_serie(serie_out, series)
            loss = loss_class + 0.5 * loss_serie
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, class_pred = torch.max(class_out, 1)
            _, serie_pred = torch.max(serie_out, 1)
            
            train_total += labels.size(0)
            train_class_correct += (class_pred == labels).sum().item()
            train_serie_correct += (serie_pred == series).sum().item()
        
        train_acc = 100.0 * train_class_correct / train_total
        train_serie_acc = 100.0 * train_serie_correct / train_total
        
        model.eval()
        val_class_correct = 0
        val_serie_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels, series in val_loader:
                images, labels, series = images.to(device), labels.to(device), series.to(device)
                class_out, serie_out = model(images)
                _, class_pred = torch.max(class_out, 1)
                _, serie_pred = torch.max(serie_out, 1)
                val_total += labels.size(0)
                val_class_correct += (class_pred == labels).sum().item()
                val_serie_correct += (serie_pred == series).sum().item()
        
        val_acc = 100.0 * val_class_correct / val_total
        val_serie_acc = 100.0 * val_serie_correct / val_total
        
        scheduler.step()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {train_loss/len(train_loader):.4f}")
        print(f"  Train - Clase: {train_acc:.1f}% | Serie: {train_serie_acc:.1f}%")
        print(f"  Val   - Clase: {val_acc:.1f}% | Serie: {val_serie_acc:.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': BILL_CLASSES,
                'series': SERIES,
                'img_size': IMG_SIZE
            }, 'billete_model.pth')
            print(f"  -> Modelo guardado (best val acc: {best_val_acc:.2f}%)")
    
    print(f"\n{'=' * 50}")
    print(f"ENTRENAMIENTO COMPLETADO!")
    print(f"Mejor accuracy de clase: {best_val_acc:.2f}%")
    print(f"Modelo guardado en: billete_model.pth")
    print(f"{'=' * 50}")
    return model


if __name__ == '__main__':
    dataset_path = 'dataset'
    
    if not Path(dataset_path).exists():
        print(f"Error: No se encontró la carpeta '{dataset_path}'")
        exit(1)
    
    train_model(dataset_path, epochs=50, batch_size=4, learning_rate=0.0005)
