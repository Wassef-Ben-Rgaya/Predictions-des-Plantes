# 🤖 Smart Greenhouse — Plant Prediction

<div align="center">

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![YOLOv12](https://img.shields.io/badge/YOLOv12-00FFFF?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-92--95%25-brightgreen?style=for-the-badge)

Part of the [🌿 Smart Greenhouse System](https://github.com/Wassef-Ben-Rgaya/smart-greenhouse)

</div>

---

## 📖 Description

AI/ML module for plant detection, classification, and growth stage prediction. Trained on a custom dataset of greenhouse crops (Spinach, Romaine Lettuce, Radish) with Train/Val/Test splits. Deployed via a Dockerized Flask REST API on Render.

---

## 🤖 Models

| Model | Task | Accuracy |
|-------|------|----------|
| YOLOv12 | Plant detection & localization | 92–95% |
| MobileNetV2 | Plant species classification | ~93% |
| CNN (custom) | Growth stage prediction | ~92% |

---

## 🌱 Supported Plant Classes

| Class | Description |
|-------|-------------|
| 🥬 Romaine Lettuce | Detection & growth stage tracking |
| 🌿 Spinach | Detection & growth stage tracking |
| 🌱 Radish | Detection & growth stage tracking |

---

## 🏗️ Project Structure

```
smart-greenhouse-plant-prediction/
├── app/                 # Flask application
├── models/              # Model architecture definitions
├── weights/             # Trained model weights (.h5, .pt)
├── main.py              # Application entry point
├── requirements.txt     # Python dependencies
├── Dockerfile           # Docker configuration
├── Procfile             # Render/Heroku deployment config
├── runtime.txt          # Python runtime version
├── .dockerignore        # Docker ignored files
└── .gitignore           # Git ignored files
```

---

## 🚀 Installation

### Option A — Local (Python)

```bash
# 1. Clone the repository
git clone https://github.com/Wassef-Ben-Rgaya/smart-greenhouse-plant-prediction.git
cd smart-greenhouse-plant-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Flask API
python main.py
```

### Option B — Docker

```bash
# Build the image
docker build -t smart-greenhouse-plant-prediction .

# Run the container
docker run -p 5000:5000 smart-greenhouse-plant-prediction
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Classify plant species from image |
| POST | `/api/detect` | Detect & locate plants (YOLOv12) |
| GET | `/api/growth` | Get plant growth stage prediction |
| GET | `/api/health` | API health check |

### Example Request

```bash
curl -X POST https://your-render-url/api/predict \
  -F "image=@plant.jpg"
```

### Example Response

```json
{
  "plant": "Romaine Lettuce",
  "confidence": 0.94,
  "growth_stage": "Early",
  "recommendation": "Increase watering frequency"
}
```

---

## 📊 Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| YOLOv12 | 95% | 0.94 | 0.93 | 0.935 |
| MobileNetV2 | 93% | 0.92 | 0.91 | 0.915 |
| CNN (custom) | 92% | 0.91 | 0.90 | 0.905 |

---

## ☁️ Deployment

This API is deployed on **Render** using Docker.

```
Dockerfile   → Container build
Procfile     → Process definition for Render
runtime.txt  → Python version specification
```

---

## 🔗 Related Repositories

| Repo | Description |
|------|-------------|
| [smart-greenhouse-backend](https://github.com/Wassef-Ben-Rgaya/smart-greenhouse-backend) | Node.js REST API |
| [smart-greenhouse-mobile](https://github.com/Wassef-Ben-Rgaya/smart-greenhouse-mobile) | Flutter mobile app |
| [smart-greenhouse-iot](https://github.com/Wassef-Ben-Rgaya/smart-greenhouse-iot) | Raspberry Pi IoT scripts |

---

## 👨‍💻 Author

**Wassef BEN RGAYA** — [LinkedIn](https://www.linkedin.com/in/wassef-ben-rgaya-600817188) · [GitHub](https://github.com/Wassef-Ben-Rgaya)

© 2025 — Polytech Tunis Final Year Project
