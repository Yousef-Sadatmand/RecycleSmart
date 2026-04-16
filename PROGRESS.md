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

### [x] Write train.py with fixed data pipeline
- 70/15/15 stratified split (split first, augment after)
- Augmentation on training set only
- class_weight for trash imbalance (weight: 3.069)
- Working early stopping watching val_loss
- MobileNetV2 base frozen, custom classification head
- **Phase 1 result: 82.6% test accuracy**
- Fine-tuning phase added (unfroze top 54 layers, lr=1e-4)
- **Phase 2 result: 85.5% test accuracy**
- Saved to: models/mobilenetv2_finetuned.h5

---

## Current Issue
- Overfitting: train accuracy ~98%, val/test ~85%
- Root cause: small dataset (1,768 training images across 6 classes)
- Fix planned: stronger regularization (more dropout + more aggressive augmentation)

---

## Up Next

### [ ] Improve regularization in train.py
- Increase dropout 0.3 → 0.5
- Add rotation + zoom augmentation
- Target: 90%+ test accuracy

### [ ] Retrain and validate — target 90%+ on held-out test set

### [ ] Gradio web app for browser-based testing

### [ ] Convert model to TFLite

### [ ] Build mobile app (React Native or Flutter)
