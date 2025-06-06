# Garbage Classification and Detection using Deep Learning

  This project focuses on classifying and detecting different types of garbage-**cardboard, paper, metal, plastic, and glass**-in images using deep learning models.It combines **image classification** (to identify the types of trash an object is) and **object detection** (to localize each trash item in a photo) using TensorFlow and YOLO.

## Objectives

  - Classify types of garbage in single-item images.
  - Detect and classify trash items in real-world, mixed garbage scenes.
  - Contribute to waste sorting and recycling automation efforts.

## Software used

  - Python
  - TensorFlow
  - YOLOv5
  - OpenCV
  - CVAT
  - Docker
  - VS Code
  - Github

## How to use

1. Clone the repo
     ```bash
     git clone https://github.com/PhooSoanHan/ML-Project.git
     cd garbage-classification
2. Create a virtual environment
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
3. Install Dependencies
     ```bash
     pip install -r requirements.txt
4. Train Classifier
     ```bash
     python scripts/train_classifier.py
5. Test Classifier
     ```bash
     python scripts/test_model.py
6. Run CVAT
     ```bash
     cd cvat
     docker compose up -d
