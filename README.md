# 🚁 Real-Time Drone Detection System

A lightweight, real-time **Computer Vision system** designed to detect **drones, airplanes, helicopters, and birds** in images and live video streams.  
The model is trained using the **TensorFlow Object Detection API** and deployed as a highly optimized **TensorFlow Lite (TFLite)** model for fast inference on **CPU and edge devices** (Windows / Raspberry Pi).

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Model Architecture](#-model-architecture)
- [Dataset & Classes](#-dataset--classes)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training Process & Challenges](#-training-process--challenges)
- [Results](#-results)
- [Credits](#-credits)

---

## 🔍 Project Overview

Unauthorized drones pose serious risks in **restricted and sensitive airspace**.  
This project addresses that problem by using **transfer learning** to repurpose a pre-trained object detection network to identify aerial objects accurately and efficiently.

### ✨ Key Features
- **Real-Time Detection:** Runs smoothly on standard CPUs using TFLite
- **Multi-Class Detection:** Detects and classifies:
  - Drones
  - Birds
  - Airplanes
  - Helicopters
- **Dual Mode Operation:**
  - Live **Webcam Detection**
  - **Batch Image Processing** from a folder
- **Edge Optimized:** Suitable for low-resource devices

---

## 🧠 Model Architecture

- **Base Model:** `SSD MobileNet V2 FPNLite 320x320`

### Why this model?
- **SSD (Single Shot Detector):**
  - Performs detection in a single forward pass
  - Extremely fast → ideal for real-time video
- **MobileNet V2:**
  - Lightweight architecture
  - Optimized for mobile and edge devices
- **FPNLite (Feature Pyramid Network):**
  - Improves detection of **small and distant objects** like drones
  - Uses multi-scale feature extraction

---

## 📂 Dataset & Classes

The model was trained on a **custom-labeled dataset (~500 images)**.

### Dataset Split
- **Training:** 80%
- **Validation:** 20%

### Classes
1. `drone`
2. `airplane`
3. `helicopter`
4. `bird`

### Negative Mining
To reduce false positives, the dataset included **background-only images** (empty skies, clouds, buildings without aerial objects).  
This helped the model learn when **not** to detect anything.

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Drone-Detection-System.git
cd Drone-Detection-System
2️⃣ Install Dependencies
Make sure Python is installed, then run:

bash
Copy code
pip install tensorflow opencv-python numpy
⚠️ Raspberry Pi users:
Install tflite-runtime instead of full TensorFlow for better performance.

3️⃣ Folder Structure
Ensure your project directory looks like this:

text
Copy code
├── drone_detector.tflite      # Trained TFLite model
├── labelmap.txt               # Class labels
├── run_detection.py           # Main inference script
├── test_images/               # (Optional) Input images
└── output_images/             # (Auto-generated) Output results
🚀 Usage
Run the detection script:

bash
Copy code
python run_detection.py
You will be prompted to choose a mode.

🎥 Mode 1: Live Webcam Detection
Uses your system webcam (Camera ID 0)

Displays bounding boxes and confidence scores in real time

Press q to exit

🖼️ Mode 2: Image Folder Processing
Reads all images from the test_images/ folder

Saves annotated images to output_images/

Supports .jpg, .jpeg, .png, .bmp

🛠 Training Process & Challenges
The model was trained on Kaggle GPUs using a custom TensorFlow Object Detection pipeline.
Several non-trivial engineering challenges were encountered and resolved.

1️⃣ Dependency Management (Dependency Hell)
Challenge:

TensorFlow Object Detection API depends on:

TensorFlow 2.x

Protobuf 3.x

Modern environments ship with incompatible versions (TF 2.16+, Protobuf 4.x)

Solution:

Created a controlled environment:

tensorflow==2.15.0

protobuf==3.20.3

Patched legacy issues such as:

tf.case vs control_flow_ops.case

Deprecated tf-slim syntax

2️⃣ Training Instability (NaN Loss)
Challenge:

Default learning rate (0.8) with small batch size caused NaN loss

Model weights exploded during training

Solution:

Carefully tuned hyperparameters:

Batch Size: 4 (GPU memory constraint)

Learning Rate: 0.004

Resulted in stable convergence

3️⃣ Data Engineering Issues
Challenge:

XML annotation parsing errors

Filenames contained spaces and mixed extensions (.jpg, .png)

Solution:

Wrote a preprocessing script to:

Sanitize filenames

Normalize extensions

Validate bounding boxes

Ensured clean TFRecord generation

📊 Results
Training Steps: 5000

Final Total Loss: ~0.28 (excellent convergence)

Inference Speed: ~30–60 FPS on standard CPU using TFLite

Detection Quality: Strong performance on small and distant aerial objects

🤝 Credits
Developer: Nithaesh Raja

Tech Stack: Python, TensorFlow, TensorFlow Lite, OpenCV

Training Platform: Kaggle Kernels

Inference Platform: Windows / Edge Devices
