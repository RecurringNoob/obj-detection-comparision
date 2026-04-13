#  Object Detection Pipeline

A modular, object‑oriented Python project for detecting vehicles and other objects in traffic images using **Faster R‑CNN**, **YOLOv5**, and **YOLOv8**. The pipeline is split into reusable modules for data preprocessing, model inference, and post‑processing/visualization.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modules Explained](#modules-explained)
- [Customization](#customization)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## ✨ Features

- **Multiple Model Support**  
  Seamlessly switch between:
  - Faster R‑CNN (ResNet‑50 FPN) from Torchvision
  - YOLOv5 and YOLOv8 via Ultralytics
  
- **Modular OOP Design**  
  Clean separation of concerns:
  - `data_preprocessing.py` – Image loading and tensor conversion
  - `model_inference.py` – Abstract model interface and concrete implementations
  - `post_processing.py` – Bounding box drawing, frequency analysis, threshold plots
  - `utils.py` – Shared visualization helpers

- **Comprehensive Post‑Processing**  
  - Filter detections by confidence threshold
  - Count class frequencies
  - Generate per‑class frequency vs. threshold plots
  - Visualise total detection count across thresholds

- **Easy to Extend**  
  Add new models by subclassing `DetectionModel`.

## 📁 Project Structure
traffic_detection/
├── main.py # Orchestrator script
├── data_preprocessing.py # ImagePreprocessor class
├── model_inference.py # FasterRCNNModel, YOLOModel classes
├── post_processing.py # DetectionPostProcessor class
├── utils.py # show_image() helper
├── requirements.txt # Python dependencies
├── bangalore-traffic-jam.jpg # Sample image (not included)
└── README.md # This file

text

## ⚙️ Installation

1. **Clone the repository** (or download the source code).

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate      # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
Note: The first time you run a YOLO model, Ultralytics will automatically download the weights (e.g., yolov8s.pt).
```
4. **▶️ Usage** 
Place your input image in the project folder and update the IMAGE_PATH variable inside main.py if necessary.

5.**Run the main script:**

```bash
python main.py
```
By default, the script will:

Run Faster R‑CNN inference

Run YOLOv5 (or YOLOv8) inference

Display annotated images

Print detection frequencies

Show frequency vs. threshold plots for Faster R‑CNN

To run only one model, simply comment out the unwanted function call in main().

# 🧩 Modules Explained
data_preprocessing.py
ImagePreprocessor
Loads an image from disk, provides BGR/RGB arrays, and converts to a PyTorch tensor with optional batch dimension.

model_inference.py
DetectionModel (abstract base class) – defines predict() interface.

FasterRCNNModel – loads a pre‑trained Faster R‑CNN and returns {boxes, labels, scores}.

YOLOModel – wrapper for Ultralytics YOLO (v5/v8). predict() returns the native Ultralytics Results object.

post_processing.py
DetectionPostProcessor

draw_boxes() – annotates image with bounding boxes and counts frequencies.

compute_frequency_over_thresholds() – builds frequency matrices for threshold analysis.

plot_class_frequency() / plot_total_detections() – generate matplotlib plots.

utils.py
show_image() – converts BGR to RGB and displays using matplotlib.

🛠️ Customization
Use a Different Image
Change the IMAGE_PATH variable in main.py.

Change Model Confidence Threshold
Modify the score_threshold parameter when calling post.draw_boxes().

Add a New Detection Model
Subclass DetectionModel in model_inference.py.

Implement predict().

Instantiate and call it in main.py.

Adjust Threshold Analysis Range
Edit the THRESHOLDS list in main.py (e.g., [0.1, 0.2, ...]).

# 🖼️ Example Output
Faster R‑CNN annotated image
https://screenshots/faster_rcnn_output.png (example placeholder)

Frequency vs Threshold Plot
https://screenshots/frequency_plot.png (example placeholder)

Terminal output:

text
=== Faster R-CNN ===
Detection frequencies (conf > 0.4): {'car': 15, 'truck': 3, 'bus': 2, ...}
...
=== YOLO (yolov5s.pt) ===
Class: car, Conf: 0.87, Box: [234, 198, 456, 320]
...
# 📦 Dependencies
See requirements.txt for exact versions.
Core libraries:

torch & torchvision – Faster R‑CNN

ultralytics – YOLOv5/v8

opencv-python – image handling and drawing

matplotlib – plotting

numpy – numerical operations

# 📄 License
This project is licensed under the MIT License – see the LICENSE file for details.

# 🙏 Acknowledgements
COCO dataset for pre‑trained model weights.

Ultralytics for the excellent YOLO implementation.

PyTorch team for Torchvision models.
