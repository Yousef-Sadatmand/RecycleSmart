"""
evaluate.py — RecycleSmart model evaluation

Loads the best trained model and runs it against the held-out test set.
Outputs:
  - Per-class precision, recall, F1 score
  - Confusion matrix (saved as evaluation/confusion_matrix.png)
  - Overall test accuracy
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = "Data/merged"
MODEL_PATH = "models/efficientnetb0_9class_finetuned.keras"
OUTPUT_DIR = "evaluation"
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

CLASS_NAMES = ["battery", "biological", "cardboard", "glass", "metal",
               "paper", "plastic", "textiles", "trash"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 1. Rebuild the same test split as train.py ────────────────────────────────
# We use the same SEED and split ratios so we get the exact same 380 test images.

all_paths  = []
all_labels = []

for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(class_name)

sort_idx   = np.argsort(all_paths)
all_paths  = np.array(all_paths)[sort_idx]
all_labels = np.array(all_labels)[sort_idx]

le = LabelEncoder()
le.fit(CLASS_NAMES)
all_labels_enc = le.transform(all_labels)

_, X_temp, _, y_temp = train_test_split(
    all_paths, all_labels_enc,
    test_size=0.30, stratify=all_labels_enc, random_state=SEED
)
_, X_test, _, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"Test set size: {len(X_test)} images")

# ── 2. Build test dataset ─────────────────────────────────────────────────────

def load_and_preprocess(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

AUTOTUNE = tf.data.AUTOTUNE
test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

# ── 3. Load model ─────────────────────────────────────────────────────────────

print(f"Loading model from {MODEL_PATH}…")
model = tf.keras.models.load_model(MODEL_PATH)

# ── 4. Get predictions ────────────────────────────────────────────────────────
# model.predict returns probabilities for each class (6 numbers per image).
# np.argmax picks the index of the highest probability = predicted class.

print("Running predictions…")
y_prob = model.predict(test_ds, verbose=1)   # shape: (380, 6)
y_pred = np.argmax(y_prob, axis=1)           # shape: (380,)

# ── 5. Classification report ──────────────────────────────────────────────────
# precision = of all times it said "glass", how often was it right?
# recall    = of all actual glass images, how many did it catch?
# F1        = harmonic mean of precision and recall (overall class score)

print("\n── Classification Report ──\n")
report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
print(report)

# Save report to file
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# ── 6. Confusion matrix ───────────────────────────────────────────────────────
# Rows = actual class, Columns = predicted class.
# Diagonal = correct predictions. Off-diagonal = mistakes.
# e.g. row "plastic", column "glass" = how many plastic images were called glass.

cm = confusion_matrix(y_test, y_pred)

# Normalize to percentages so all classes are comparable
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm_norm,
    annot=True,
    fmt=".0%",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax
)
ax.set_xlabel("Predicted", fontsize=12)
ax.set_ylabel("Actual", fontsize=12)
ax.set_title("Confusion Matrix — RecycleSmart MobileNetV2 (fine-tuned)", fontsize=13)
plt.tight_layout()

output_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
plt.savefig(output_path, dpi=150)
print(f"\nConfusion matrix saved to: {output_path}")

# ── 7. Overall accuracy ───────────────────────────────────────────────────────

overall_acc = (y_pred == y_test).mean()
print(f"Overall test accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
