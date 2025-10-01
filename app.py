import gradio as gr
from ultralytics import YOLO
from PIL import Image
import cv2

# ========================
# Load Trained YOLO Model
# ========================
model = YOLO("best.pt")  # Make sure best.pt is in the same folder

# ========================
# Detection Functions
# ========================

# For image detection
def detect_image(img):
    results = model.predict(img)
    annotated_image = results[0].plot()
    return Image.fromarray(annotated_image)

# For video detection
def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    output_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        annotated_frame = results[0].plot()
        output_frames.append(annotated_frame)

    cap.release()

    # Save output video
    out_path = "output.mp4"
    if output_frames:
        height, width, _ = output_frames[0].shape
        out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
        for f in output_frames:
            out.write(f)
        out.release()
    return out_path

# ========================
# Gradio UI
# ========================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ♻️ Garbage Classification with YOLOv8  
        Upload an **image** or **video** to classify garbage into categories:  
        🟦 Cardboard | 🟩 Glass | 🟨 Metal | 📄 Paper | 🟧 Plastic | 🗑️ Trash  
        """
    )

    with gr.Tab("📷 Image Detection"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label=None, show_label=False)
                image_button = gr.Button("Run Detection")
            with gr.Column():
                image_output = gr.Image(type="pil", label=None, show_label=False)

        image_button.click(fn=detect_image, inputs=image_input, outputs=image_output)

    with gr.Tab("🎥 Video Detection"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label=None, show_label=False)
                video_button = gr.Button("Run Detection")
            with gr.Column():
                video_output = gr.Video(label=None, show_label=False)

        video_button.click(fn=detect_video, inputs=video_input, outputs=video_output)

# ========================
# Run App
# ========================
if __name__ == "__main__":
    demo.launch()
