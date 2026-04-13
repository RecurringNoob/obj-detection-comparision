import cv2
import torch
from torchvision.transforms import functional as F

class ImagePreprocessor:
    """Load and preprocess images for object detection models."""

    def __init__(self, image_path: str):
        self.image_path = image_path
        self.img_bgr = None
        self.img_rgb = None
        self.img_tensor = None
        self._load_image()

    def _load_image(self):
        """Load image from disk in both BGR and RGB formats."""
        self.img_bgr = cv2.imread(self.image_path)
        if self.img_bgr is None:
            raise FileNotFoundError(f"Could not load image at {self.image_path}")
        self.img_rgb = cv2.cvtColor(self.img_bgr, cv2.COLOR_BGR2RGB)

    def get_bgr(self):
        return self.img_bgr.copy()

    def get_rgb(self):
        return self.img_rgb.copy()

    def to_tensor(self, add_batch_dim=True):
        """Convert RGB image to tensor. Optionally add batch dimension."""
        tensor = F.to_tensor(self.img_rgb)
        if add_batch_dim:
            tensor = tensor.unsqueeze(0)  # shape: [1, 3, H, W]
        self.img_tensor = tensor
        return tensor