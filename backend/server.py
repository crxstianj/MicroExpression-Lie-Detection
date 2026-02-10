import os
import uuid
import cv2
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
from model import LieDetectorCNN
from PIL import Image
from collections import Counter

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cambia esto en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LieDetectorCNN().to(device)
model.load_state_dict(torch.load("models/custom_lie_detector.pth", map_location=device))
model.eval()

# Transformación
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
LABELS = ["Truth", "Lie"]

def predict_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.argmax(output, dim=1).item()
    return LABELS[pred]

@app.post("/predict_video/")
async def predict_from_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        return {"error": "Invalid file format"}

    contents = await file.read()
    temp_name = f"{uuid.uuid4()}.mp4"
    with open(temp_name, "wb") as f:
        f.write(contents)

    cap = cv2.VideoCapture(temp_name)
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        label = predict_frame(frame)
        predictions.append(label)

    cap.release()
    os.remove(temp_name)

    counts = Counter(predictions)
    final = counts.most_common(1)[0][0] if counts else "Unknown"

    return {
        "Truth": counts["Truth"],
        "Lie": counts["Lie"],
        "FinalPrediction": final
    }
