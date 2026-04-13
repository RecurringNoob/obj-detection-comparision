from data_preprocessing import ImagePreprocessor
from model_inference import FasterRCNNModel, YOLOModel, COCO_CLASSES
from post_processing import DetectionPostProcessor
from utils import show_image

# Configuration
IMAGE_PATH = "bangalore-traffic-jam.jpg"
THRESHOLDS = [round(t, 1) for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]

def run_faster_rcnn_pipeline():
    print("\n=== Faster R-CNN ===")
    # 1. Preprocessing
    preprocessor = ImagePreprocessor(IMAGE_PATH)
    img_tensor = preprocessor.to_tensor(add_batch_dim=True)

    # 2. Model inference
    model = FasterRCNNModel()
    detections = model.predict(img_tensor)  # dict with boxes, labels, scores

    # 3. Post-processing
    post = DetectionPostProcessor(COCO_CLASSES)
    annotated_img, freq = post.draw_boxes(preprocessor.get_bgr(), detections, score_threshold=0.4)
    show_image(annotated_img, "Faster R-CNN Detections (conf > 0.4)")
    print("Detection frequencies (conf > 0.4):", freq)

    # Threshold analysis
    all_freq, all_classes, freq_matrix = post.compute_frequency_over_thresholds(
        detections, COCO_CLASSES, THRESHOLDS
    )
    post.plot_class_frequency(THRESHOLDS, all_classes, freq_matrix)
    post.plot_total_detections(THRESHOLDS, all_freq)

def run_yolo_pipeline(model_name="yolov8s.pt"):
    print(f"\n=== YOLO ({model_name}) ===")
    # 1. Preprocessing (YOLO can accept image path directly)
    preprocessor = ImagePreprocessor(IMAGE_PATH)

    # 2. Model inference
    model = YOLOModel(model_name)
    result = model.predict(IMAGE_PATH)  # or preprocessor.get_bgr()

    # 3. Post-processing (plotting built into Ultralytics)
    annotated_img = result.plot()
    show_image(annotated_img, f"{model_name.upper()} Detections")

    # Print detection details
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist()
            print(f"Class: {COCO_CLASSES[cls_id]}, Conf: {conf:.2f}, Box: {xyxy}")

def main():
    # Choose which model to run (you can run both)
    run_faster_rcnn_pipeline()
    run_yolo_pipeline("yolov5s.pt")   # or "yolov8s.pt"

if __name__ == "__main__":
    main()