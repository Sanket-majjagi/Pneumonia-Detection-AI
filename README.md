# 🩺 Pneumonia Detection using MobileNetV2

## Overview
This project detects Pneumonia from Chest X-Ray images using a fine-tuned MobileNetV2 deep learning model, with Grad-CAM visualization to show which regions of the X-ray influenced the prediction.

## Features
- Transfer Learning using MobileNetV2
- Class imbalance handling with class weights
- Tkinter GUI for interactive image prediction
- Grad-CAM visualization (model attention heatmap overlay)
- Tuned decision threshold (optimized for sensitivity/specificity trade-off)
- Full performance evaluation (Accuracy, Confusion Matrix, ROC Curve)

## Results
Evaluated on a held-out test set of 624 chest X-ray images:

| Metric | Score |
|---|---|
| Accuracy | 87.66% |
| ROC-AUC | 0.960 |
| Precision (Pneumonia) | 85.33% |
| Recall / Sensitivity (Pneumonia) | 96.92% |
| F1-Score (Pneumonia) | 90.76% |
| Chosen decision threshold | 0.836 |

Confusion matrix and ROC curve are included in the repository (`confusion_matrix.png.png`, `roc_curve.png.png`).

> Sensitivity is intentionally prioritized over specificity, since missing a true pneumonia case is more costly than a false alarm in a screening context.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy, OpenCV
- Matplotlib
- Scikit-learn
- Tkinter (GUI)

## How to Run

1. Clone the repo and install dependencies:
pip install -r requirements.txt


2. Run the GUI:

python pneumonia_gui.py

3. Click **"Select Chest X-ray Image"**, choose a JPEG/PNG chest X-ray, and view the prediction with confidence score. Toggle **Grad-CAM** to see which regions of the image influenced the model's decision.

## Dataset
Trained on the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset (Kaggle). Not included in this repo due to size — download separately and place in a `chest_xray/` folder if you want to retrain.

## Disclaimer
This tool is for educational/research purposes only and is **not** a certified medical diagnostic device. Predictions should not be used as a substitute for professional medical evaluation.
