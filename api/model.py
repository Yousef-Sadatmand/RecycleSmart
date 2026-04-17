"""
model.py — Model loading and inference for RecycleSmart API

Loads the EfficientNetB0 model once at startup and exposes a single
predict() function that the API calls for every request.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Path is relative to the project root, where uvicorn is launched from
MODEL_PATH = Path("models/efficientnetb0_9class_finetuned.keras")
IMG_SIZE   = (224, 224)

CLASS_NAMES = ["battery", "biological", "cardboard", "glass", "metal",
               "paper", "plastic", "textiles", "trash"]

BIN_INSTRUCTIONS = {
    "battery":    "⚠️ Hazardous waste — take to a battery drop-off depot. Never put in any bin.",
    "biological": "🟤 Compost/organics bin — food scraps and food-soiled paper belong here.",
    "cardboard":  "♻️ Recycling bin — flatten and remove any tape or staples first.",
    "glass":      "🟢 Glass recycling bin — rinse it out, remove the lid.",
    "metal":      "♻️ Recycling bin — rinse cans, remove labels if possible.",
    "paper":      "♻️ Recycling bin — keep dry. Soiled paper goes in compost.",
    "plastic":    "♻️ Recycling bin — check the number. #1 and #2 are widely accepted.",
    "textiles":   "👕 Donation bin or textile recycling depot — do not put in any curbside bin.",
    "trash":      "🗑️ General waste bin — this item cannot be recycled.",
}

CONFIDENCE_THRESHOLD = 0.70   # below this → warn the user to check local guidelines

# ── Load model once at startup ────────────────────────────────────────────────
# Loading a TensorFlow model takes ~3 seconds. We do it once here so every
# request after that is fast. The API imports this module on startup.

print(f"Loading model from {MODEL_PATH}…")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model ready.")

# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes (JPEG or PNG from the HTTP request),
    runs the model, and returns a result dict.

    Returns:
        {
            "class":           "plastic",
            "confidence":      0.94,
            "bin_instruction": "♻️ Recycling bin...",
            "low_confidence":  False,
            "all_scores":      {"battery": 0.01, ..., "plastic": 0.94, ...}
        }
    """
    # Decode bytes → tensor
    img = tf.image.decode_image(
        tf.constant(image_bytes), channels=3, expand_animations=False
    )
    img.set_shape([None, None, 3])

    # Resize + preprocess — must match train.py exactly
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)   # add batch dimension: (1, 224, 224, 3)

    # Run inference
    scores = model.predict(img, verbose=0)[0]   # shape: (9,)

    top_idx    = int(np.argmax(scores))
    top_class  = CLASS_NAMES[top_idx]
    confidence = float(scores[top_idx])

    return {
        "class":           top_class,
        "confidence":      round(confidence, 4),
        "bin_instruction": BIN_INSTRUCTIONS[top_class],
        "low_confidence":  confidence < CONFIDENCE_THRESHOLD,
        "all_scores":      {CLASS_NAMES[i]: round(float(scores[i]), 4)
                            for i in range(len(CLASS_NAMES))},
    }
