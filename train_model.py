# train_model.py
from ultralytics import YOLO

DATA_YAML = "./dataset/data.yaml"   # Path to your data.yaml
MODEL_NAME = "yolov8n.pt"           # Pretrained YOLOv8 model
EPOCHS = 10

# Load YOLOv8 model
model = YOLO(MODEL_NAME)

# Train the model
model.train(
    data=DATA_YAML,
    epochs=EPOCHS,
    imgsz=640,
    device=0  # set to 0 for GPU, -1 for CPU
)

print("✅ Training finished. Best weights are saved in runs/detect/train/weights/best.pt")
