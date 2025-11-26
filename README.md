📌 Problem Statement

Astronauts depend on safety equipment—oxygen tanks, fire extinguishers, emergency phones, etc.
If any of these are missing, misplaced, or obscured, it can lead to life-threatening consequences.

The goal is to build an AI model that can:

Detect all 7 safety objects

Work reliably under harsh environmental variations

Produce accurate metrics and a safety readiness score

Provide interpretable visualizations for auditing

🎯 Objectives

Train YOLOv9-E for multi-class safety equipment detection

Integrate attention mechanisms to improve detection accuracy

Simulate space-like distortions through custom augmentations

Generate complete evaluation metrics (mAP, confusion matrix, precision/recall, failures)

Provide prediction and dataset visualization tools

Make the project runnable in Google Colab / Kaggle

🧠 Features
✔ YOLOv9-E Training Pipeline

Using custom hyperparameters and optimizers (AdamW, Mosaic, lr scheduling)

✔ Attention-Enhanced Architecture

CBAM or SE blocks integrated into backbone for higher robustness

✔ Space-Condition Augmentation Engine

Simulates:

Low light

Harsh shadows

Motion blur

Zero-gravity rotations

Occlusions

Noise & distortions

✔ Evaluation & Reporting

Automatically generates:

mAP@0.5

mAP@0.5:0.95

Precision & Recall

F1 Score

Confusion Matrix

PR Curve

F1 Curve

Training loss curves

Failure case analysis

✔ Prediction & Visualization Tools

predict.py — inference on test images

visualize.py — view dataset label bounding boxes
