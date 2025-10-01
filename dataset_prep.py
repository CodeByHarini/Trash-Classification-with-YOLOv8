import os
import random
import shutil
from PIL import Image

# Paths
SOURCE_DIR = "./Garbage classification"  # Folder with your original images
DEST_DIR = "./dataset"                    # YOLO formatted dataset

SPLITS = [("train", 0.7), ("valid", 0.2), ("test", 0.1)]
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Create YOLO folders
for split, _ in SPLITS:
    os.makedirs(os.path.join(DEST_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, split, "labels"), exist_ok=True)

# Helper function to create dummy YOLO labels (full image)
def create_yolo_label(img_path, class_id, label_path):
    try:
        img = Image.open(img_path)
        w, h = img.size
        x_center, y_center = 0.5, 0.5
        width, height = 1.0, 1.0
        with open(label_path, "w") as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
    except:
        pass

# Process dataset
for class_id, class_name in enumerate(CLASSES):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.exists(class_dir):
        continue
    images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    n_total = len(images)
    n_train = int(SPLITS[0][1]*n_total)
    n_valid = int(SPLITS[1][1]*n_total)
    splits_data = {
        "train": images[:n_train],
        "valid": images[n_train:n_train+n_valid],
        "test": images[n_train+n_valid:]
    }
    for split, split_images in splits_data.items():
        for img_file in split_images:
            src_img_path = os.path.join(class_dir, img_file)
            dst_img_path = os.path.join(DEST_DIR, split, "images", img_file)
            dst_label_path = os.path.join(DEST_DIR, split, "labels", os.path.splitext(img_file)[0]+".txt")
            shutil.copy(src_img_path, dst_img_path)
            create_yolo_label(src_img_path, class_id, dst_label_path)

# Create data.yaml
data_yaml = f"""
train: {os.path.join(DEST_DIR, 'train/images')}
val: {os.path.join(DEST_DIR, 'valid/images')}
test: {os.path.join(DEST_DIR, 'test/images')}

nc: {len(CLASSES)}
names: {CLASSES}
"""

with open(os.path.join(DEST_DIR, "data.yaml"), "w") as f:
    f.write(data_yaml)

print("✅ Dataset converted to YOLO format with train/valid/test and data.yaml")
