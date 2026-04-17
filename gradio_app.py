"""
gradio_app.py — RecycleSmart browser demo

Upload a photo of any waste item and get:
  - Predicted category (cardboard, glass, metal, paper, plastic, trash)
  - Confidence score
  - Which bin to put it in
  - Confidence bar chart for all 6 classes

Run in Colab:
  !pip install gradio -q
  !python gradio_app.py
"""

import numpy as np
import tensorflow as tf
import gradio as gr

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH  = "models/efficientnetb0_9class_finetuned.keras"
IMG_SIZE    = (224, 224)
CLASS_NAMES = ["battery", "biological", "cardboard", "glass", "metal",
               "paper", "plastic", "textiles", "trash"]

# Bin instructions shown to the user for each class
BIN_INSTRUCTIONS = {
    "battery":    "⚠️  Hazardous waste — take to a battery drop-off depot. Never put in any bin.",
    "biological": "🟤  Compost/organics bin — food scraps and food-soiled paper belong here.",
    "cardboard":  "♻️  Recycling bin — flatten and remove any tape or staples first.",
    "glass":      "🟢  Glass recycling bin — rinse it out, remove the lid.",
    "metal":      "♻️  Recycling bin — rinse cans, remove labels if possible.",
    "paper":      "♻️  Recycling bin — keep dry. Soiled paper goes in compost.",
    "plastic":    "♻️  Recycling bin — check the number. #1 and #2 are widely accepted.",
    "textiles":   "👕  Donation bin or textile recycling depot — do not put in any curbside bin.",
    "trash":      "🗑️  General waste bin — this item cannot be recycled.",
}

# ── Load model ────────────────────────────────────────────────────────────────

print(f"Loading model from {MODEL_PATH}…")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model ready.")

# ── Prediction function ───────────────────────────────────────────────────────

def predict(image):
    """
    Takes a PIL image from Gradio, returns:
      - label string with bin instructions
      - dict of {class: confidence} for the bar chart
    """
    # Resize and preprocess — must match train.py exactly
    img = tf.image.resize(image, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)   # add batch dimension: (1, 224, 224, 3)

    # Run inference
    predictions = model.predict(img, verbose=0)[0]   # shape: (6,)

    # Top prediction
    top_idx        = int(np.argmax(predictions))
    top_class      = CLASS_NAMES[top_idx]
    top_confidence = float(predictions[top_idx])

    # Label shown above the bar chart
    label = (
        f"Prediction: {top_class.upper()}  ({top_confidence*100:.1f}% confident)\n\n"
        f"{BIN_INSTRUCTIONS[top_class]}"
    )

    # Bar chart data — dict of {class_name: probability}
    confidences = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}

    return label, confidences

# ── Gradio UI ─────────────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy", label="Upload a photo of your waste item"),
    outputs=[
        gr.Textbox(label="Result & Bin Instructions", lines=3),
        gr.Label(num_top_classes=9, label="Confidence by class"),
    ],
    title="RecycleSmart ♻️",
    description=(
        "Point your camera at a piece of waste and find out which bin it belongs in. "
        "Powered by EfficientNetB0 trained on 14,786 images across 9 waste categories."
    ),
    examples=[],
    theme=gr.themes.Soft(),
)

demo.launch(share=True)   # share=True creates a public link valid for 72 hours
