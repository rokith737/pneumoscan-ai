🩺 PneumoScan AI

AI-Powered Pneumonia Detection from Chest X-Rays with Full-Stack Deployment

📌 Description

PneumoScan AI is an end-to-end medical AI system designed to detect pneumonia from chest X-ray images using deep learning. It integrates a ResNet-18 model, FastAPI backend, NVIDIA Triton inference server, and a React frontend, along with LLM-generated clinical reports.

This project bridges the gap between research and real-world deployment by providing a complete, reproducible, and scalable solution.

🚀 Features
🔍 Pneumonia detection from chest X-rays
⚡ Fast GPU inference using Triton Server
🧠 AI-generated clinical reports (GPT-based)
🌐 Full-stack web application (React + FastAPI)
🐳 Docker-based one-command deployment
📊 Confidence scores & probability visualization
📄 Structured medical output
🏗️ Architecture
User → React Frontend → FastAPI Backend → Triton Server → Model (ONNX)
                                            ↓
                                     GPT Report API
🧠 Model Information
Model: ResNet-18 (Transfer Learning)
Dataset: NIH Chest X-ray Dataset
Task: Binary Classification (Normal / Pneumonia)
Accuracy: ROC-AUC ≈ 0.92+
Inference Time: ~15–30 ms (GPU)
🔄 Workflow
Upload chest X-ray image
Image preprocessing
Model inference via Triton
Probability calculation
Report generation using LLM
Display results in UI
🖥️ Tech Stack
Backend
Python
FastAPI
PyTorch
ONNX
Frontend
React
Vite
Deployment
Docker
NVIDIA Triton Server
AI Integration
GPT-based report generation
⚙️ Requirements
Python 3.11
Node.js 20
Docker & Docker Compose
NVIDIA GPU (recommended)
CUDA 12.x
🐳 Installation
git clone https://github.com/rokith737/pneumoscan-ai.git
cd pneumoscan-ai
docker compose up --build
🌐 Access
Frontend: http://localhost:3000
Backend: http://localhost:8080
API Docs: http://localhost:8080/docs
📡 API
POST /predict

Upload chest X-ray image and get:

Prediction
Confidence
Probabilities
Inference time
AI-generated report
📊 Example Output
Prediction: PNEUMONIA
Confidence: 93.2%
P(Normal): 6.8%
P(Pneumonia): 93.2%
📉 Limitations
Only binary classification
No visual explanation (Grad-CAM not included)
Dataset-specific training
LLM dependency
🔮 Future Work
Multi-disease classification
Explainable AI (Grad-CAM)
CPU optimization
Real-world clinical validation
📄 License

MIT License

👨‍💻 Author

Rokith S
📧 rokith737shanmugam@gmail.com
