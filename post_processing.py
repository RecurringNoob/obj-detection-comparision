import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from utils import show_image

class DetectionPostProcessor:
    """Post‑processing: drawing, filtering, and analysis of detections."""

    def __init__(self, class_names):
        self.class_names = class_names

    def draw_boxes(self, image_bgr, detections, score_threshold=0.4, color=(0,255,0)):
        """
        Draw bounding boxes and labels on a copy of the image.
        detections: dict with 'boxes', 'labels', 'scores' (tensors or lists)
        Returns: annotated image, frequency dict
        """
        img_draw = image_bgr.copy()
        freq = {}

        boxes = detections["boxes"]
        labels = detections["labels"]
        scores = detections["scores"]

        for box, label, score in zip(boxes, labels, scores):
            if score < score_threshold:
                continue
            x1, y1, x2, y2 = map(int, box.tolist() if hasattr(box, 'tolist') else box)
            cls_name = self.class_names[label]
            freq[cls_name] = freq.get(cls_name, 0) + 1

            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            text = f"{cls_name} {score:.2f}"
            cv2.putText(img_draw, text, (x1, max(y1-5,0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        return img_draw, freq

    @staticmethod
    def compute_frequency_over_thresholds(detections, class_names, thresholds):
        """
        For each threshold, count class frequencies.
        Returns: dict {threshold: Counter}, and a matrix for plotting.
        """
        labels = detections["labels"]
        scores = detections["scores"]
        all_freq = {}

        for th in thresholds:
            mask = scores > th
            filtered_labels = labels[mask]
            cls_names = [class_names[lbl] for lbl in filtered_labels.tolist()]
            all_freq[th] = Counter(cls_names)

        # Build matrix for plotting
        all_classes = sorted({cls for freq in all_freq.values() for cls in freq.keys()})
        freq_matrix = []
        for th in thresholds:
            row = [all_freq[th].get(cls, 0) for cls in all_classes]
            freq_matrix.append(row)
        freq_matrix = np.array(freq_matrix)

        return all_freq, all_classes, freq_matrix

    @staticmethod
    def plot_class_frequency(thresholds, all_classes, freq_matrix):
        plt.figure(figsize=(14, 7))
        for i, cls in enumerate(all_classes):
            plt.plot(thresholds, freq_matrix[:, i], marker="o", linewidth=2, label=cls)
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Frequency of Detections")
        plt.title("Per-Class Detection Frequency Across Thresholds")
        plt.xticks(thresholds)
        plt.grid(alpha=0.3)
        plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_total_detections(thresholds, all_freq):
        total_counts = {th: sum(freq.values()) for th, freq in all_freq.items()}
        plt.figure(figsize=(8,5))
        plt.plot(list(total_counts.keys()), list(total_counts.values()), marker='o')
        plt.title("Total Detections vs Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Total Objects Detected")
        plt.grid(True)
        plt.show()