import torch
import torchvision
from ultralytics import YOLO
from abc import ABC, abstractmethod

# COCO class names (used by Faster R-CNN and YOLO)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class DetectionModel(ABC):
    """Abstract base class for detection models."""
    @abstractmethod
    def predict(self, image):
        """Run inference and return detections."""
        pass

class FasterRCNNModel(DetectionModel):
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        )
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_tensor):
        """image_tensor: [1, C, H, W] tensor"""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            outputs = self.model(image_tensor)[0]
        # Move outputs to CPU for further processing
        boxes = outputs["boxes"].cpu()
        labels = outputs["labels"].cpu()
        scores = outputs["scores"].cpu()
        return {"boxes": boxes, "labels": labels, "scores": scores}

class YOLOModel(DetectionModel):
    """Wrapper for YOLOv5 and YOLOv8 (Ultralytics)"""
    def __init__(self, model_name="yolov8s.pt"):
        self.model = YOLO(model_name)  # supports yolov5s.pt, yolov8s.pt, etc.

    def predict(self, image):
        """image can be path (str) or numpy array (BGR)"""
        results = self.model.predict(image, verbose=False)
        # Return the first result (for single image)
        return results[0]