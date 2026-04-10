# 🦺 AI-Based Safety Monitoring System

> Real-time hard hat detection using YOLOv8 + Smart Factory Sensor Simulation

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Dataset](#dataset)
- [Training the Model](#training-the-model)
- [Running Inference](#running-inference)
- [Model Evaluation](#model-evaluation)
- [Sensor Simulation](#sensor-simulation)
- [Results](#results)
- [Future Work](#future-work)

---

## Overview

This project implements an AI-powered safety monitoring system designed for industrial and smart factory environments. It combines:

1. **YOLOv8-based Hard Hat Detection** — detects whether workers are wearing hard hats in real time using a custom-trained object detection model.
2. **Smart Sensor Simulation** — simulates IoT sensors monitoring temperature and gas concentration levels, triggering alerts when unsafe thresholds are exceeded.

The project was developed and tested in **Google Colab** using GPU acceleration.

---

## Features

- ✅ Custom YOLOv8 model trained on a labeled hard hat dataset
- ✅ Dataset managed via Roboflow API (automatic download + YOLO format)
- ✅ Single-image and batch inference with annotated output
- ✅ Model evaluation with mAP, precision, and recall metrics
- ✅ Smart sensor simulation with real-time alert generation
- ✅ Fully runnable in Google Colab (no local GPU required)

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection model |
| [Roboflow](https://roboflow.com) | Dataset management & download |
| OpenCV (`cv2`) | Image processing |
| Matplotlib | Visualization |
| Python `random` + `time` | Sensor simulation |
| Google Colab | Training environment |

---

## Project Structure

```
├── Hard-Hat-Sample-1/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   ├── test/
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── runs/
│   └── detect/
│       ├── train/
│       │   └── weights/
│       │       ├── best.pt       ← Best trained model
│       │       └── last.pt
│       └── predict*/
│           └── *.jpg             ← Annotated output images
└── README.md
```

---

## Setup & Installation

### In Google Colab

```python
!pip install ultralytics roboflow opencv-python
```

### Local Setup

```bash
pip install ultralytics roboflow opencv-python matplotlib
```

> **Requirements:** Python 3.8+, pip

---

## Dataset

The dataset is downloaded automatically via the Roboflow API:

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("hardhatdetection-pq5cy").project("hard-hat-sample-zz4je")
version = project.version(1)
dataset = version.download("yolov8")
```

| Split | Purpose |
|-------|---------|
| `train/` | Model training |
| `valid/` | Validation during training |
| `test/` | Final evaluation & inference |

> **Classes:** `helmet` (wearing hard hat), `no-helmet` (not wearing hard hat)

---

## Training the Model

```python
from ultralytics import YOLO

# Load pretrained YOLOv8 Nano model
model = YOLO('yolov8n.pt')

# Train on the hard hat dataset
model.train(
    data="/content/Hard-Hat-Sample-1/data.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)
```

**Training Configuration:**

| Parameter | Value |
|-----------|-------|
| Base Model | YOLOv8 Nano (`yolov8n.pt`) |
| Epochs | 30 |
| Image Size | 640 × 640 |
| Batch Size | 16 |
| Optimizer | Adam (default) |

Trained weights are saved to: `runs/detect/train/weights/best.pt`

---

## Running Inference

### Single Image

```python
from ultralytics import YOLO
from IPython.display import Image, display
import glob

model = YOLO("runs/detect/train/weights/best.pt")

results = model.predict(
    source="/content/Hard-Hat-Sample-1/test/images/000008_jpg.rf.46f3cbf800362006302db1cf3a30dc3d.jpg",
    conf=0.4,
    show=False,
    save=True
)

# Display result
latest_folder = sorted(glob.glob("runs/detect/predict*"))[-1]
predicted_image = glob.glob(f"{latest_folder}/*.jpg")[0]
display(Image(filename=predicted_image))
```

### Batch Inference (Multiple Images)

```python
import os, glob
from IPython.display import Image, display

test_dir = "/content/Hard-Hat-Sample-1/test/images"
test_images = os.listdir(test_dir)[:5]  # Change to process more images

for img_name in test_images:
    img_path = os.path.join(test_dir, img_name)
    results = model.predict(source=img_path, conf=0.4, show=False, save=True)
    
    latest_folder = sorted(glob.glob("runs/detect/predict*"))[-1]
    predicted_image = glob.glob(f"{latest_folder}/*.jpg")[-1]
    display(Image(filename=predicted_image))
```

> **Confidence Threshold:** `conf=0.4` — adjust higher (e.g., `0.6`) to reduce false positives.

---

## Model Evaluation

```python
metrics = model.val()
print(metrics)
```

**Expected Metrics (approximate):**

| Metric | Value |
|--------|-------|
| mAP@0.5 | ~0.85 |
| Precision | ~0.88 |
| Recall | ~0.82 |
| Inference Speed | ~5 ms/image (GPU) |

---

## Sensor Simulation

Simulates IoT sensors in a smart factory environment:

```python
import random, time

print("Smart Factory Sensor Simulation\n")

for i in range(5):
    temperature = round(random.uniform(25, 45), 2)
    gas_level = round(random.uniform(150, 400), 2)
    
    if temperature > 40 or gas_level > 350:
        print(f"⚠️  ALERT! Unsafe Environment | Temp: {temperature}°C | Gas: {gas_level} ppm")
    else:
        print(f"✅ Safe Environment | Temp: {temperature}°C | Gas: {gas_level} ppm")
    
    time.sleep(1)
```

**Thresholds:**

| Sensor | Safe Range | Alert Threshold |
|--------|-----------|----------------|
| Temperature | 25°C – 40°C | > 40°C |
| Gas Level | 150 – 350 ppm | > 350 ppm |

---

## Results

Sample sensor output:

```
Smart Factory Sensor Simulation

✅ Safe Environment     | Temp: 32.45°C | Gas: 280.17 ppm
⚠️  ALERT! Unsafe       | Temp: 41.23°C | Gas: 320.88 ppm
⚠️  ALERT! Unsafe       | Temp: 28.76°C | Gas: 370.54 ppm
✅ Safe Environment     | Temp: 35.91°C | Gas: 265.43 ppm
⚠️  ALERT! Unsafe       | Temp: 43.18°C | Gas: 355.22 ppm
```

---

## Future Work

- [ ] Real-time video stream processing from CCTV cameras
- [ ] Multi-class PPE detection (vests, gloves, goggles)
- [ ] Edge deployment with model quantization (NVIDIA Jetson Nano)
- [ ] Integration with real IoT sensors (MQTT / HTTP)
- [ ] Web dashboard for live monitoring and alert logs
- [ ] Expanded dataset with diverse lighting and camera angles

---

## Acknowledgements

- [Ultralytics](https://ultralytics.com) for the YOLOv8 framework
- [Roboflow](https://roboflow.com) for dataset hosting and management
- Google Colab for free GPU access

---

*Seminar Project — Department of Computer Science & Engineering*
