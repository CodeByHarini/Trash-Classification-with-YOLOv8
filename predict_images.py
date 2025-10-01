# predict_images.py
from ultralytics import YOLO
from PIL import Image
import os

# Path to your trained YOLOv8 weights
MODEL_PATH = "./runs/detect/train/weights/best.pt"

# Load YOLOv8 model
model = YOLO("best.pt")

# Folder where test images are located
TEST_IMAGES_DIR = "./test_images"
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)

# Predict on all images in the folder
for img_file in os.listdir(TEST_IMAGES_DIR):
    if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(TEST_IMAGES_DIR, img_file)
        results = model.predict(img_path)
        # Save annotated image
        annotated_img = results[0].plot()
        save_path = os.path.join(TEST_IMAGES_DIR, f"pred_{img_file}")
        Image.fromarray(annotated_img).save(save_path)
        print(f"Saved prediction: {save_path}")
