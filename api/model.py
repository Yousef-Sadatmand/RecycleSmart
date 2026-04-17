"""
model.py — Model loading and inference for RecycleSmart API

Uses TFLite runtime instead of full TensorFlow so the server fits
within Render free tier's 512MB RAM limit.

Loads the model once at startup; all subsequent requests are fast.
"""

import io
import numpy as np
from pathlib import Path
from PIL import Image

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH = Path("models/efficientnetb0_9class.tflite")
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

CONFIDENCE_THRESHOLD = 0.70

# ── Load model once at startup ────────────────────────────────────────────────

print(f"Loading TFLite model from {MODEL_PATH}…")
interpreter = Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Model ready.")

# ── Inference ─────────────────────────────────────────────────────────────────

def predict(image_bytes: bytes) -> dict:
    """
    Takes raw image bytes, runs TFLite inference, returns classification result.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)

    # EfficientNet preprocessing: scale [0, 255] → [-1, 1]
    img_array = np.array(img, dtype=np.float32)
    img_array = (img_array / 127.5) - 1.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    scores = interpreter.get_tensor(output_details[0]['index'])[0]

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
