import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import LieTruthDataset
from model import LieDetectorCNN
import os
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import numpy as np

# Ruido gaussiano
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)

# Transformaciones
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.05)
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Rutas
train_path = 'data/train/train'
test_path = 'data/test/test'

# Dataset y DataLoader
train_dataset = LieTruthDataset(train_path, transform=train_transform)
test_dataset = LieTruthDataset(test_path, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# Dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelo, pérdida y optimizador
model = LieDetectorCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Variables para gráficas y métricas
epochs = 10
train_losses = []
test_accuracies = []
all_preds = []
all_labels = []

for epoch in range(epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    end_time = time.time()
    epoch_duration = end_time - start_time

    print(f"[{epoch+1}/{epochs}] Loss: {avg_loss:.4f} | Tiempo epoch: {epoch_duration:.2f} segundos")

# Evaluación final (después de todo el entrenamiento)
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Matriz de confusión en porcentaje
cm = confusion_matrix(all_labels, all_preds)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

labels_names = ['Truth', 'Lie']

print("\nMatriz de Confusión (porcentaje):")
print(f"{'':10s}" + "".join([f"{name:>10s}" for name in labels_names]))
for i, row in enumerate(cm_percent):
    print(f"{labels_names[i]:10s}" + "".join([f"{v:10.2f}%" for v in row]))

# Guardar modelo
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/custom_lie_detector.pth")

# Graficar pérdida y exactitud (opcional)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, marker='o', label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.title("Train Loss")
plt.legend()

plt.tight_layout()
plt.show()
