import cv2
import matplotlib.pyplot as plt

def show_image(img_bgr, title="Result", figsize=(10, 8)):
    """Display a BGR image using matplotlib."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()