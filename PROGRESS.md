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

### [x] gradio_app.py — browser demo
- Image upload → predicted class + confidence + bin instructions
- Confidence bar chart for all 6 classes
- Tested on real household items — works well on target classes
- Known limitation: electronics and out-of-distribution items misclassified
  (no e-waste class yet — by design, will be fixed in dataset expansion)

---

## Roadmap

### Phase 1 — Dataset Expansion (Foundation)
**Goal: 15-20 classes that cover what people actually get wrong**
- [ ] Find and download The Garbage Dataset (GD, ~13,000 images, released 2026)
      — preferred over TACO for household use case (indoor backgrounds match app context)
- [ ] Use TACO as a supplement only (outdoor litter photos, diverse backgrounds)
- [ ] Merge GD + TrashNet (current dataset)
- [ ] Target classes to add: electronics/e-waste, food waste/organics,
      textiles, hazardous (batteries/paint), styrofoam, soft plastics
- [ ] Retrain EfficientNetB0 on expanded dataset
- [ ] Target: 90%+ accuracy on new class set
- [ ] Confidence threshold: below 70% → show "unsure, check local guidelines"
- Why first: a weak model kills the product — everything else sits on this

### Phase 2 — Web App + API Layer (Validate with Real Users)
**Goal: working product on phone browser + API municipalities can integrate into their apps**
- [ ] Build React frontend with camera capture
- [ ] Build Python/FastAPI backend serving the model
- [ ] Municipality rules layer — PostgreSQL + JSONB column for bin rules
      (rules are volatile — cities change policies, a lookup table won't scale)
- [ ] User-contributed corrections — flag wrong predictions per city
      (this data becomes the competitive moat)
- [ ] Deploy to web (Render / Railway / Fly.io)
- [ ] Expose public API so municipalities can embed in their existing city apps
      (solves the user adoption gap — meet users where they already are)
- Why API matters: selling to GFL/Recology means integrating into their apps,
  not convincing people to download a new one

### Phase 3 — Mobile App
**Goal: native iOS + Android app**
- [ ] convert_tflite.py — export model to TFLite with Dynamic Range Quantization
- [ ] Build React Native app (recommended over Flutter — shares code with web app,
      larger talent pool, strong TFLite support via JSI in 2026)
- [ ] Real-time camera feed (not just photo upload)
- [ ] Offline support — model runs on device, no internet needed
- [ ] Temporal smoothing: only update classification if same label appears
      in 3 of last 5 frames — prevents flickering during real-time scanning
- [ ] Submit to App Store + Google Play

### Phase 4 — Production & Sales
**Goal: a live product with data to show potential buyers**
- [ ] Analytics — track what items people scan most, where confusion happens
- [ ] User feedback loop — wrong predictions feed back into retraining
- [ ] Identify pilot municipality (leverage GFL industry contacts)
- [ ] Approach GFL, Waste Management, Recology with usage data + accuracy numbers
- [ ] Explore EPR (Extended Producer Responsibility) angle — in 2026 many regions
      require manufacturers to fund packaging lifecycle. App becomes compliance tool.

---

## Known Limitations (Current 6-Class Model)
- No electronics class → baby monitors, phones, cables misclassified as paper/plastic
- No food waste class → organic items misclassified
- No hazardous class → batteries, paint cans have no correct bin
- Trained on clean studio images → real-world phone photos in bad lighting may score lower
