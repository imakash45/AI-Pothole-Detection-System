# 🚧 AI Pothole Detection System

An AI-powered real-time pothole detection system built with YOLOv8 and Streamlit. 
The system detects, classifies, and analyzes potholes from images, videos, 
and live webcam feeds with intelligent road condition assessment.

---

## 🌐 Live Demo
🔗 [Click here to view the app](#) <!-- Replace with your Streamlit URL after deployment -->

---

## 📸 Screenshots

| Home Screen | Detection Result |
|---|---|
| ![Home](screenshots/home.png) | ![Demo](screenshots/demo.png) |

---

## ✨ Features

- 🖼️ **Multiple Input Modes** — Image, Video, and Live Webcam support
- 🔍 **Real-time Pothole Detection** — Powered by custom trained YOLOv8 model
- 📊 **Pothole Classification** — Classifies each pothole as:
  - 🟢 Minor
  - 🟡 Moderate
  - 🔴 Severe
- 🔢 **Pothole Counter** — Counts total potholes detected in frame
- 🛣️ **Road Condition Assessment** — Rates overall road as Poor / Moderate / Good
- 🚗 **Speed Suggestion** — Recommends ideal driving speed based on road condition
- 🔊 **Audio Alert** — Plays alert sound when pothole is detected
- 📦 **Bounding Boxes** — Visual markers with confidence scores on detections

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| YOLOv8 (Ultralytics) | Object detection model |
| Streamlit | Web application framework |
| OpenCV | Image & video processing |
| PyTorch | Deep learning backend |
| Custom Dataset | Self-annotated pothole dataset |

---

## 🧠 Model Details

- **Architecture** → YOLOv8
- **Training** → Custom trained on self-annotated road dataset
- **Dataset** → Contains labeled images of minor, moderate and severe potholes
- **Output** → Bounding boxes with classification and confidence score

---

## 📂 Project Structure

AI_POTHOLE_DETECTION/
├── app/
│   ├── app1.py          # Main Streamlit application
│   └── detection.py     # YOLOv8 detection logic
├── assets/
│   └── alert.wav        # Audio alert sound
├── screenshots/
│   ├── home.png         # App homepage screenshot
│   └── demo.png         # Detection result screenshot
├── best.pt              # Trained YOLOv8 model weights
├── requirements.txt     # Project dependencies
└── .gitignore           # Git ignore rules

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/imakash45/AI-Pothole-Detection-System.git
cd AI-Pothole-Detection-System
```

**2. Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the app**
```bash
streamlit run app/app1.py
```

---

## 👨‍💻 Developer

**Akash Kumar**
- 🐙 GitHub → [imakash45](https://github.com/imakash45)

---

## 📄 License
This project is for educational purposes as part of an AI college project.