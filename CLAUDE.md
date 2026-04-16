# RecycleSmart — Claude Project Context

## Project Goal

A mobile app (iOS/Android) where users point their phone camera at an item and get instantclassification with bin instructions and a confidence score. Built on a MobileNetV2 transferlearning model trained to classify waste into 6 recycling categories.

## My Background

* Completed BrainStation Data Science Diploma (November 2023)
* Work at GFL Environmental (waste management) — inspired this project
* Comfortable with Python; not yet experienced in deployment

* * *

## Development Environment

* IDE: VS Code (local, Windows)
* Project root: Y:\Data Science\Develop\RecycleSmart\
* OS: Windows (Y:\ drive)
* GPU training: Google Colab (2 hrs/day available)
* Python environment: .venv (virtual environment, Y:\Data Science\Develop\RecycleSmart\.venv)
* Python version: 3.11.4
* TensorFlow version: 2.21.0

* * *

## Dataset

* Source: Gary Thung's Garbage Classification dataset (Kaggle)
* Total images: 2,527
* Location: data/raw/
* Classes and counts:
  * paper: 594
  * glass: 501
  * plastic: 482
  * metal: 410
  * cardboard: 403
  * trash: 137
* Class imbalance: trash is significantly underrepresented — use class_weight in model.fit()

* * *

## Models (Capstone, 2023)

1. Logistic Regression — ~45% test accuracy, heavy overfitting (baseline)
2. Custom CNN (64x64, 2 conv layers) — ~70% test accuracy
3. MobileNetV2 transfer learning (224x224) — ~93% train accuracy (best, target for deployment)
  * Saved as: models/Recyclable_Material_Classification_MobileNetV2.h5

* * *

## Known Issues with Original Capstone Code

1. Data leakage — augmentation was applied before the train/test split
2. No validation set — only train/test, no val split during training
3. Broken early stopping — monitored val_loss but no validation_data passed to model.fit()
4. Class imbalance not handled — no class weighting for trash

* * *

## Current Next Step

Write train.py with a fixed data pipeline:

* 70/15/15 stratified train/val/test split (split FIRST, augment AFTER)
* Augmentation applied to training set only
* class_weight passed to model.fit() to handle trash imbalance
* Working early stopping with validation_data passed to model.fit()
* MobileNetV2 base, frozen layers, custom classification head

* * *

## Roadmap (in order)

* [x] Logistic Regression baseline (capstone)
* [x] Custom CNN (capstone)
* [x] MobileNetV2 transfer learning — webcam demo (capstone)
* [ ] Write train.py with fixed data pipeline
* [ ] Retrain and validate — target 90%+ on held-out test set
* [ ] Gradio web app for browser-based testing
* [ ] Convert model to TFLite
* [ ] Build mobile app (React Native or Flutter)

* * *

## File Naming Conventions

* train.py — model training script (fixed pipeline)
* predict.py — single image inference
* evaluate.py — metrics, confusion matrix
* convert_tflite.py — export model to TFLite for mobile

* * *

## Project Structure

    RecycleSmart/
    ├── CLAUDE.md
    ├── README.md
    ├── train.py
    ├── predict.py
    ├── evaluate.py
    ├── convert_tflite.py
    ├── data/
    │   └── raw/               # 6 class folders (cardboard, glass, metal, paper, plastic, trash)
    ├── models/
    │   └── Recyclable_Material_Classification_MobileNetV2.h5
    ├── notebooks/
    │   ├── model2_cnn_original.ipynb
    │   └── model3_mobilenetv2_original.ipynb
    └── docs/
        └── executive_summary.pdf

* * *

## Tech Stack

* Language: Python
* ML Framework: TensorFlow / Keras
* Base Model: MobileNetV2 (pretrained on ImageNet)
* Training: Google Colab (GPU)
* Local dev: VS Code
* Planned deployment: TFLite → React Native or Flutter