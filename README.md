# Trash Classification with YOLOv8

![Python](https://img.shields.io/badge/Python-3.12-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-orange)
![License](https://img.shields.io/badge/License-MIT-green)
[![Hugging Face Spaces](https://img.shields.io/badge/Demo-HuggingFace-yellow)](https://huggingface.co/spaces/CodeByHarini/garbage-classification-yolov8)

This repository demonstrates a deep learning–based system for **automatic trash classification** using **YOLOv8**. The model is trained to detect and classify different categories of waste from images and videos, supporting better recycling and waste management.


## 🚀 Project Overview

Classifying waste manually is time-consuming and error-prone. This project provides:

* **Real-time detection** of trash categories using YOLOv8.
* **Image & video support** with easy-to-use scripts.
* **Pretrained weights** (`best.pt`) for direct inference.
* A **Gradio web app** for interactive predictions.


## 📂 Folder Structure

```
trash-classification-with-yolov8/
│
├── dataset/                     # Training and validation dataset  
├── predictions/                 # Model prediction outputs  
├── test_images/                 # Sample test images  
│
├── app.py                       # Gradio app for demo  
├── train_model.py               # Script to train YOLOv8 model  
├── predict_images.py             # Script to run inference on images  
├── dataset_prep.py              # Prepares dataset for YOLOv8 format  
│
├── best.pt                      # Trained YOLOv8 model weights  
├── yolov8n.pt                   # Pretrained YOLOv8 base model  
├── requirements.txt             # Dependencies  
├── .gitignore                   # Ignore unnecessary files  
└── README.md                    # Project documentation  
```


## 🎥 Live Demo

Try the model on Hugging Face Spaces:
🔗 [Trash Classification YOLOv8 (Demo)](https://huggingface.co/spaces/CodeByHarini/garbage-classification-yolov8)

## 📸 Demo Screenshot / Video

### Screenshot

![Chatbot Screenshot](https://github.com/CodeByHarini/Harini-Portfolio-chatbot/blob/main/Sample%20Output.jpg)

### Video

[![Watch Video](assets/chatbot_demo_thumbnail.png)](https://github.com/CodeByHarini/Harini-Portfolio-chatbot/blob/main/Demo%20Video.mp4)




## ⚙️ Installation (Local Testing)

### 1. Clone the repository

```bash
git clone https://github.com/CodeByHarini/trash-classification-with-yolov8.git
cd trash-classification-with-yolov8
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run inference

For single/multiple images:

```bash
python predict_images.py
```

For interactive Gradio app:

```bash
python app.py
```


## 🧑‍💻 Technologies & Libraries

* **Python 3.12**
* **YOLOv8 (Ultralytics)** – Object detection & classification
* **PyTorch** – Deep learning framework
* **Gradio** – Web app for interactive demo
* **OpenCV** – Image preprocessing and visualization


## 🔮 Next Steps / Improvements

* Expand dataset to cover more trash categories.
* Deploy on mobile/edge devices for smart waste bins.
* Add real-time video stream support.
* Compare YOLOv8 results with other detection models.


## 🙏 Acknowledgements

* **Ultralytics YOLOv8** – for providing the state-of-the-art object detection framework.
* **Hugging Face Spaces** – for free model hosting and deployment.
* **Gradio** – for building the interactive web app.
* **OpenAI & ChatGPT** – for guidance in structuring project documentation.



## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.


