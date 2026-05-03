🩺 PneumoScan AI – Pneumonia Detection System

An end-to-end AI-powered medical imaging system for detecting pneumonia from chest X-rays using deep learning, deployed with a scalable microservices architecture.

🔗 GitHub Repository:
👉 https://github.com/rokith737/pneumoscan-ai

📌 Overview

PneumoScan AI is a fully integrated, production-ready system that combines:

Deep Learning (ResNet-18)
GPU-based inference (NVIDIA Triton)
FastAPI backend
React frontend
LLM-generated diagnostic reports (GPT-4o)

The system enables users to upload chest X-rays and receive:

Pneumonia prediction (Normal / Pneumonia)
Confidence score
Probability distribution
Inference latency
AI-generated clinical report

🚀 Key Features
✅ Deep Learning Model – ResNet-18 with transfer learning
⚡ Fast Inference – < 500 ms end-to-end latency
🧠 LLM Integration – Converts predictions into clinical reports
🌐 Full Stack Application – React + FastAPI
🐳 Dockerized Deployment – One-command setup
🔁 Scalable Architecture – Microservices with Triton Inference Server
📊 High Performance – ROC-AUC > 0.92
🏗️ System Architecture

The system follows a 4-layer microservices architecture:

Frontend (React) – User interface for image upload & results
Backend (FastAPI) – Orchestrates preprocessing & inference
Inference Server (Triton) – GPU-based model serving
LLM API (GPT-4o) – Generates diagnostic reports

🧠 Model Details
Architecture: ResNet-18 (pretrained on ImageNet)
Task: Binary Classification (Normal vs Pneumonia)
Dataset: NIH Chest X-ray dataset
Framework: PyTorch → ONNX → Triton
Accuracy: ROC-AUC ≈ 0.92–0.94
Inference Speed: 15–30 ms (GPU)

🔄 Workflow
Upload chest X-ray image
Image preprocessing (resize, normalize)
Inference via Triton Server
Softmax probability calculation
LLM generates diagnostic report
Results displayed in UI

🖥️ Tech Stack
🔹 Backend
Python 3.11
FastAPI
Uvicorn
PyTorch
ONNX
🔹 Frontend
React 18
Vite
CSS
🔹 AI & Deployment
NVIDIA Triton Inference Server
OpenAI GPT-4o API
Docker & Docker Compose

⚙️ Requirements
OS: Windows 10/11 or Ubuntu 22.04
GPU: NVIDIA GPU (≥ 4GB VRAM recommended)
CUDA: 12.x
Docker + NVIDIA Container Toolkit
Python 3.11
Node.js 20

🐳 Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/rokith737/pneumoscan-ai.git
cd pneumoscan-ai
2️⃣ Run with Docker
docker compose up --build
3️⃣ Access Application
Frontend: http://localhost:3000
Backend API: http://localhost:8080
Swagger Docs: http://localhost:8080/docs

📡 API Endpoints
Endpoint	Method	Description
/predict	POST	Upload image & get prediction
/health	GET	Check system health
/model/info	GET	Model metadata

📊 Example Output
Prediction: PNEUMONIA
Confidence: 93.2%
Inference Time: ~420 ms
Report: AI-generated clinical explanation

📉 Limitations
Binary classification only (Normal vs Pneumonia)
No explainability (Grad-CAM not implemented)
Trained on single dataset (NIH)
LLM dependency may raise privacy concerns

🔮 Future Improvements
Multi-label classification (14 diseases)
Grad-CAM visual explanations
CPU optimization (INT8 models)
Multi-hospital validation

License
MIT License

 Author
 Rokith S
 rokith737shanmugam@gmail.com

Acknowledgments
NIH Chest X-ray Dataset
NVIDIA Triton Inference Server
OpenAI GPT-4o API
