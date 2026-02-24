# 🔍 AI-Powered Face & Text-Guided Detection System

A Streamlit-based real-time detection system that combines:

- 👤 Face recognition (InsightFace)
- 🎯 Object detection (YOLOv8)
- 🔍 Text-guided detection (OWL-ViT)
- 🧠 Similarity-based matching (CLIP)
- 📹 Live camera, video, and image support

---

## 🚀 Features

- Real-time face recognition with reference images
- Object detection for people, bags, laptops, etc.
- Natural language text-guided detection
- Live camera detection using WebRTC
- Video processing with annotated output
- Detection logs, analytics, and exports (CSV / JSON)

---

## 🧠 Tech Stack

- **Frontend:** Streamlit
- **Face Recognition:** InsightFace (ArcFace)
- **Object Detection:** YOLOv8 (Ultralytics)
- **Text-Guided Detection:** OWL-ViT, CLIP (Hugging Face)
- **Video:** OpenCV, streamlit-webrtc
- **Backend:** Python, NumPy, PyTorch

---

## 🖥️ System Requirements

- **Python:** 3.9 or 3.10 (recommended)
- **OS:** Windows / Linux / macOS
- **RAM:** 8 GB minimum (16 GB recommended)

⚠️ **Windows users must install Microsoft Visual C++ Redistributable (x64)**  
👉 https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist

---

## ⚙️ Installation (Conda – recommended)

```bash
conda create -n faceenv python=3.10
conda activate faceenv
pip install -r requirements.txt
