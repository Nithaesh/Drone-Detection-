# 🚁 Real-Time Drone Detection System

A lightweight, real-time Computer Vision system designed to detect **drones, airplanes, helicopters, and birds** in images and live video streams.  
The model is trained using the **TensorFlow Object Detection API** and deployed as a highly optimized **TensorFlow Lite (TFLite)** model for fast inference on CPUs and edge devices (Windows / Raspberry Pi).

---

## 📖 Project Overview

Unauthorized drones pose serious risks in restricted and sensitive airspace.  
This project addresses that problem using **transfer learning**, repurposing a pre-trained object detection network to accurately identify aerial objects in real time.

### Key Features
- Real-time detection using TensorFlow Lite
- Multi-class classification (Drone, Bird, Airplane, Helicopter)
- Live webcam inference
- Batch image processing
- Edge-device friendly and CPU optimized

---

## 🧠 Model Architecture

- **Base Model:** SSD MobileNet V2 FPNLite (320×320)

### Architecture Rationale
- **SSD (Single Shot Detector):** Enables fast, single-pass object detection
- **MobileNet V2:** Lightweight and efficient for low-power devices
- **FPNLite:** Improves detection of small and distant objects like drones using multi-scale feature maps

---

## 📂 Dataset & Classes

The model was trained on a custom dataset consisting of approximately **500 annotated images**.

### Dataset Split
- Training: 80%
- Validation: 20%

### Classes
1. drone  
2. airplane  
3. helicopter  
4. bird  

### Negative Mining
Background-only images (empty skies, clouds, buildings without aerial objects) were included to reduce false positives and improve generalization.

---

## ⚙️ Installation

### Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Drone-Detection-System.git
cd Drone-Detection-System
```

### Install Dependencies
```bash
pip install tensorflow opencv-python numpy
```

> Raspberry Pi users should install `tflite-runtime` instead of full TensorFlow for better performance.

---

## 📁 Project Structure

```
├── drone_detector.tflite      # Trained TFLite model
├── labelmap.txt               # Class labels
├── run_detection.py           # Main inference script
├── test_images/               # Input images (optional)
└── output_images/             # Detection results
```

---

## 🚀 Usage

Run the detection system:
```bash
python run_detection.py
```

### Mode 1: Live Webcam Detection
- Uses the default system camera
- Displays bounding boxes and confidence scores in real time
- Press `q` to exit

### Mode 2: Image Folder Processing
- Reads all images from the `test_images/` directory
- Saves annotated outputs to `output_images/`
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

---

## 🛠 Training Process & Challenges

The model was trained on **Kaggle GPUs** using the TensorFlow Object Detection API.

### Dependency Management
- `tensorflow==2.15.0`
- `protobuf==3.20.3`

### Training Stability
- Batch Size: 4
- Learning Rate: 0.004

### Data Engineering
- Filename sanitization
- XML annotation normalization
- TFRecord validation

---

## 📊 Results

- Training Steps: 5000
- Final Total Loss: ~0.28
- Inference Speed: ~30–60 FPS on CPU using TFLite

---

## 🤝 Credits

- **Developer:** Nithaesh Raja  
- **Tech Stack:** Python, TensorFlow, TensorFlow Lite, OpenCV  
- **Training Platform:** Kaggle  
- **Inference Platform:** Windows / Edge Devices  

---
