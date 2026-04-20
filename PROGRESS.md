# RecycleSmart — Progress Log

## Vision
A production mobile app (iOS + Android) where users point their camera at
any household waste item and get instant bin instructions. Target market:
general public at home. Business goal: sell to large waste management
companies (GFL, Waste Management, Recology) who need to reduce recycling
contamination — a real, expensive problem municipalities already budget for.

## Solo project. Building to sell.

---

## Completed

### [x] Logistic Regression baseline (capstone)
- ~45% test accuracy, heavy overfitting

### [x] Custom CNN (capstone)
- ~70% test accuracy

### [x] MobileNetV2 transfer learning — webcam demo (capstone)
- ~93% train accuracy (inflated — data leakage, no val set, broken early stopping)

### [x] train.py — fixed pipeline + EfficientNetB0 (6-class)
- 70/15/15 stratified split, sorted paths for deterministic splits
- Augmentation on training set only (flips + brightness)
- class_weight for trash imbalance (weight: 3.069)
- Working early stopping watching val_loss
- Switched from MobileNetV2 → EfficientNetB0 (compound scaling, better accuracy)
- Phase 1: frozen base, custom head only
- Phase 2: fine-tuning top 30 layers, BN layers kept frozen, lr=1e-4
- **Final test accuracy: 92.4%** ✓ (target was 90%+)

### [x] prepare_data.py — merged dataset (TrashNet + Garbage Dataset)
- TrashNet (6 classes, 2,527 images) + Garbage Dataset (10 classes, 12,259 images)
- Output: Data/merged/ — 9 classes, 14,786 images
- New classes added: battery (756), biological (699), textiles (3,341 — shoes + clothes merged)
- shoes and clothes merged into one "textiles" class (both are donation/depot items)

### [x] train.py — 9-class EfficientNetB0 (final model)
- Dataset: 14,786 images across 9 classes
- Split: 10,350 train / 2,218 val / 2,218 test
- Class weights: battery 2.174, biological 2.352, trash 2.785 (underrepresented classes boosted)
- Phase 1 (frozen base): best epoch 6 — val_accuracy 95.72%, stopped at epoch 11
- Phase 2 (fine-tuning top 30 layers, BN frozen, lr=1e-4): best epoch 7 — val_accuracy 96.12%, stopped at epoch 12
- **Final test accuracy: 95.76%** ✓ (target was 90%+, exceeds 6-class result of 92.4%)
- Saved to: models/efficientnetb0_9class_finetuned.keras
- Exported to: models/efficientnetb0_9class_savedmodel/ (for TFLite)

### [x] evaluate.py — confusion matrix + classification report (9-class)
Per-class F1 scores (2,216 test images):
- battery:    0.99 — excellent despite being a small class
- biological: 0.99 — excellent
- textiles:   0.99 — excellent
- cardboard:  0.97
- paper:      0.96
- glass:      0.94
- metal:      0.94
- plastic:    0.94 (precision 0.92 — some non-plastic items called plastic)
- trash:      0.82 — weakest class; expected (catch-all, high visual diversity)
- Overall accuracy: 96.0% | macro avg F1: 0.95
- Note: 2 WebP files in dataset skipped (TF decode_image doesn't support WebP)

### [x] FastAPI backend — /predict + /health endpoints
- POST /predict — accepts image upload, returns class + confidence + bin instruction + all 9 scores
- GET /health — liveness check for deployment platforms
- CORS enabled for frontend
- Model loads once at startup, all subsequent requests are fast
- 70% confidence threshold — low_confidence: true triggers warning in UI
- Tested locally: cardboard photo → 99.97% confidence, correct class
- Run locally: uvicorn api.main:app --reload (from project root)

### [x] convert_tflite.py — TFLite export with Dynamic Range Quantization
- Input: models/efficientnetb0_9class_savedmodel (34.5 MB)
- Output: models/efficientnetb0_9class.tflite (4.5 MB)
- Size reduction: 87% — fits comfortably on a mobile device
- Verified: input (1, 224, 224, 3), output (1, 9), sums to 1.0
- Ready for React Native integration

### [x] Switched API to TFLite runtime for deployment
- Replaced tensorflow==2.21.0 (500MB) with ai-edge-litert (17MB) — fits Render free tier
- api/model.py rewritten: TFLite Interpreter + PIL/numpy preprocessing (no TF dependency)
- Added .python-version (3.11.9) so Render uses a supported Python version
- TFLite model (4.5MB) committed to GitHub via .gitignore exception

### [x] React frontend — real-time camera classification
- Live camera feed, no photo upload — camera opens immediately on app load
- Green scanning reticle overlay
- Frames sampled every 600ms, sent to FastAPI /predict
- Temporal smoothing: display updates only when same class appears 3 of last 5 frames
- Result panel: class name, bin type label, confidence %, bin instruction
- Top 3 confidence bars, color-coded by bin type
- Low confidence warning shown at <70%
- Color scheme: green=recycling, amber=compost, red=hazardous, purple=donate, gray=trash
- VITE_API_URL env var for easy deployment config
- Tested and working locally at localhost:5173

---

## Roadmap

### [x] Phase 2 Step 1 — Deploy to web COMPLETE
**Live URLs:**
- Backend API: https://recyclesmart-api.onrender.com
- Frontend: https://recycle-smart.vercel.app

**What was built:**
- FastAPI backend deployed to Render free tier (Python 3.11, ai-edge-litert)
- React frontend deployed to Vercel, auto-deploys on every git push to main
- Switched from full TensorFlow to TFLite for deployment (500MB → 16MB)
- Fixed critical preprocessing bug: EfficientNet in Keras 3 normalizes internally — raw [0,255] pixels passed directly
- Fixed TFLite quantization issue: reconverted without Dynamic Range Quantization
- Used absolute model path via __file__ for cross-platform compatibility
- Tested and working on phone browser ✓

---

### ← NEXT SESSION → Phase 2 Step 2 — Municipality rules layer
**Goal: bin instructions that are specific to the user's city, not generic**
- [ ] PostgreSQL database with a bin_rules table (city + class → instruction)
- [ ] Start with City of Vancouver rules hardcoded
- [ ] API updated to accept optional ?city= parameter
- [ ] User-contributed corrections — flag wrong predictions per city
      (this data becomes the competitive moat over time)

### Phase 3 — Mobile App
**Goal: native iOS + Android app**
- [ ] Build React Native app (shares logic with web app)
- [ ] Integrate TFLite model (models/efficientnetb0_9class.tflite already built)
- [ ] On-device inference — no internet needed, faster than web
- [ ] Temporal smoothing already designed — port from web frontend
- [ ] Submit to App Store + Google Play

### Phase 4 — Production & Sales
**Goal: a live product with data to show potential buyers**
- [ ] Analytics — track what items people scan most, where confusion happens
- [ ] User feedback loop — wrong predictions feed back into retraining
- [ ] Pilot municipality: City of Vancouver (first target)
- [ ] Approach GFL, Waste Management, Recology with usage data + accuracy numbers
- [ ] Expose public API so municipalities can embed in their existing city apps
- [ ] Explore EPR (Extended Producer Responsibility) angle — app becomes compliance tool

---

## Known Limitations (Current Model)
- trash class F1 0.82 — catch-all category, hardest to classify by nature
- plastic precision 0.92 — occasional false positives from other materials
- Trained on studio images — real-world phone photos in bad lighting may score lower
- No e-waste class yet — electronics misclassified (known gap, needs more data)
