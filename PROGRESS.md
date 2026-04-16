# RecycleSmart — Progress Log

## Status: In Progress

---

## Completed

### [x] Logistic Regression baseline (capstone)
- ~45% test accuracy, heavy overfitting

### [x] Custom CNN (capstone)
- ~70% test accuracy

### [x] MobileNetV2 transfer learning — webcam demo (capstone)
- ~93% train accuracy (inflated — data leakage, no val set, broken early stopping)

### [x] train.py — fixed pipeline + EfficientNetB0
- 70/15/15 stratified split, sorted paths for deterministic splits
- Augmentation on training set only (flips + brightness)
- class_weight for trash imbalance (weight: 3.069)
- Working early stopping watching val_loss
- Switched from MobileNetV2 → EfficientNetB0 (compound scaling, better accuracy)
- Phase 1: frozen base, custom head only
- Phase 2: fine-tuning top 30 layers, BN layers kept frozen, lr=1e-4
- **Final test accuracy: 92.4%** ✓ (target was 90%+)
- Saved to: models/efficientnetb0_finetuned.keras
- Exported to: models/efficientnetb0_savedmodel/ (for TFLite)

### [x] evaluate.py — confusion matrix + classification report
- Per-class precision, recall, F1
- Best classes: cardboard (96% F1), paper (94% F1)
- Weakest class: trash (87% F1) — expected given only 137 training images
- Output: evaluation/confusion_matrix.png

---

## Up Next

### [ ] Gradio web app for browser-based testing
### [ ] convert_tflite.py — export model to TFLite with quantization
### [ ] Build mobile app (React Native or Flutter)
