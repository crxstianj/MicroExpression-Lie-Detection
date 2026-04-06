# MicroExpression Lie Detection

## ¿Cómo funciona?

1. El modelo CNN se entrena con imágenes de rostros etiquetadas como `truth` o `lie`
2. Al recibir un video, la API extrae cada frame y lo clasifica individualmente
3. La predicción final se determina por **mayoría de votos** entre todos los frames analizados

## Arquitectura del modelo
```
Conv2d(3→32) → ReLU → MaxPool
Conv2d(32→64) → ReLU → MaxPool
Conv2d(64→128) → ReLU → MaxPool
Linear(128×28×28 → 256) → ReLU → Dropout(0.5)
Linear(256 → 2)  [Truth / Lie]

Entrada: imágenes RGB de 224×224. Augmentation con ruido gaussiano durante el entrenamiento.
```

## Estructura del proyecto
```
├── main.py              # Entrenamiento del modelo
├── model.py             # Arquitectura LieDetectorCNN
├── dataset.py           # Dataset personalizado
├── backend/
│   ├── server.py        # API FastAPI
│   ├── model.py         # Copia del modelo para el servidor
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   └── index.html       # Interfaz web para subir videos
└── docker-compose.yml
```

## Estructura esperada del dataset
```
data/
├── train/train/
│   ├── truth/<persona>/<pregunta>/.png
│   └── lie/<persona>/<pregunta>/.png
└── test/test/
├── truth/...
└── lie/...
```

## Uso

### Con Docker
```bash
docker-compose up --build
```

- Interfaz web: `http://localhost:8080`
- API: `http://localhost:8000`

### Local
```bash
# Entrenar
python main.py

# Levantar servidor
cd backend && uvicorn server:app --reload
```

## API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| `POST` | `/predict_video/` | Recibe un video y devuelve la predicción |

**Formatos aceptados:** `.mp4`, `.avi`, `.mov`

**Respuesta:**
```json
{
  "Truth": 142,
  "Lie": 58,
  "FinalPrediction": "Truth"
}
```

## Dependencias
```bash
pip install torch torchvision fastapi uvicorn opencv-python pillow scikit-learn matplotlib
```
