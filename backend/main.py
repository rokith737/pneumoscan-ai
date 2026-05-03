from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
from PIL import Image
import io

app = FastAPI()

# -------------------------------
# Enable CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "../training/model.pth"

model = models.resnet18(weights=None)

# Match training architecture
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)
)

state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)

model.eval()

# -------------------------------
# IMPORTANT: Correct Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(   # 🔥 THIS FIXES YOUR PROBLEM
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------
# Routes
# -------------------------------

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        image = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image)
            probs = torch.softmax(outputs, dim=1)

        normal_prob = float(probs[0][0])
        pneumonia_prob = float(probs[0][1])

        if pneumonia_prob > normal_prob:
            label = "PNEUMONIA"
            confidence = pneumonia_prob
        else:
            label = "NORMAL"
            confidence = normal_prob

        return {
            "prediction": label,
            "confidence": confidence,
            "normal_prob": normal_prob,
            "pneumonia_prob": pneumonia_prob
        }

    except Exception as e:
        return {"error": str(e)}