"""
train.py — RecycleSmart EfficientNetB0 training script (fixed pipeline)

Upgraded from MobileNetV2 to EfficientNetB0:
  - Same parameter count range but better accuracy via compound scaling
  - Better suited for small datasets than EfficientNetV2S (too large/prone to overfit)
  - Still TFLite-convertible for mobile deployment

Pipeline fixes vs. original capstone:
  1. Split FIRST (70/15/15 stratified), augment AFTER — no data leakage
  2. Validation set added — model has unseen data to tune against during training
  3. Early stopping correctly watches val_loss (validation_data passed to model.fit)
  4. class_weight passed to model.fit to handle trash class imbalance
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = "data/raw"          # 6 class folders live here
MODEL_OUT   = "models/efficientnetb0_fixed.keras"     # checkpoint during phase 1
IMG_SIZE    = (224, 224)          # EfficientNetB0 native size is 224×224
BATCH_SIZE  = 32
EPOCHS      = 50                  # early stopping will cut this short
SEED        = 42

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# ── 1. Collect file paths + labels ────────────────────────────────────────────
# Walk the data/raw folder and build two parallel lists:
#   all_paths  → ["data/raw/paper/img001.jpg", ...]
#   all_labels → ["paper", ...]

all_paths  = []
all_labels = []

for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATA_DIR, class_name)
    for fname in os.listdir(class_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(class_name)

# Sort by path to guarantee identical ordering every run — os.listdir order
# is not deterministic, so without sorting, random_state=42 produces a
# different split each time the script runs on a different session/machine.
sort_idx   = np.argsort(all_paths)
all_paths  = np.array(all_paths)[sort_idx]
all_labels = np.array(all_labels)[sort_idx]

print(f"Total images found: {len(all_paths)}")
for cn in CLASS_NAMES:
    print(f"  {cn}: {(all_labels == cn).sum()}")

# ── 2. Encode string labels → integers ───────────────────────────────────────
# Neural networks need numbers, not strings.
# LabelEncoder maps  cardboard→0, glass→1, metal→2, paper→3, plastic→4, trash→5

le = LabelEncoder()
le.fit(CLASS_NAMES)                         # fix the order to CLASS_NAMES
all_labels_enc = le.transform(all_labels)   # array of ints

# ── 3. Stratified split: 70 train / 15 val / 15 test ─────────────────────────
# "Stratified" means each split has the same class proportions as the full set.
# We do two splits:  full → train + temp(30%),  then temp → val + test (50/50)

X_train, X_temp, y_train, y_temp = train_test_split(
    all_paths, all_labels_enc,
    test_size=0.30, stratify=all_labels_enc, random_state=SEED
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50, stratify=y_temp, random_state=SEED
)

print(f"\nSplit sizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")

# ── 4. class_weight — make the model pay attention to rare classes ─────────────
# compute_class_weight returns a weight per class.
# Classes with fewer images get a higher weight so their errors cost more.

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(CLASS_NAMES)),
    y=y_train
)
class_weight_dict = dict(enumerate(class_weights_array))

print("\nClass weights:")
for i, cn in enumerate(CLASS_NAMES):
    print(f"  {cn}: {class_weight_dict[i]:.3f}")

# ── 5. Build tf.data pipelines ────────────────────────────────────────────────
# tf.data is TensorFlow's efficient data-loading system.
# It loads images from disk in parallel while the GPU trains, so there's no wait.

AUTOTUNE = tf.data.AUTOTUNE

def load_and_preprocess(path, label):
    """Read one image file, resize it. EfficientNet expects pixels in [0, 255]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.keras.applications.efficientnet.preprocess_input(img)  # scales to [-1,1]
    return img, label

def augment(img, label):
    """Random flips and brightness shift — applied to TRAINING set only."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, label

def make_dataset(paths, labels, augment_data=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED)
    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    if augment_data:
        ds = ds.map(augment, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds

# Augmentation ON for training, OFF for val/test
train_ds = make_dataset(X_train, y_train, augment_data=True,  shuffle=True)
val_ds   = make_dataset(X_val,   y_val,   augment_data=False, shuffle=False)
test_ds  = make_dataset(X_test,  y_test,  augment_data=False, shuffle=False)

# ── 6. Build the model ────────────────────────────────────────────────────────
# EfficientNetB0 pretrained on ImageNet = powerful feature extractor out of the box.
# Uses compound scaling (depth + width + resolution balanced together) which makes
# it more accurate than MobileNetV2 at a similar parameter count.
# We freeze its layers and add a new classification head for our 6 waste categories.

base_model = EfficientNetB0(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,       # drop ImageNet's 1000-class head
    weights="imagenet"
)
base_model.trainable = False  # freeze — only our head trains

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),   # flatten feature maps → 1D vector
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),               # randomly zero 30% of neurons → less overfitting
    layers.Dense(len(CLASS_NAMES), activation="softmax")  # 6-class output
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",  # labels are ints, not one-hot
    metrics=["accuracy"]
)

model.summary()

# ── 7. Callbacks ──────────────────────────────────────────────────────────────
# EarlyStopping: watches val_loss; if it doesn't improve for 5 epochs, stop
#   and restore the weights from the best epoch (restore_best_weights=True).
# ModelCheckpoint: saves the best model to disk whenever val_loss improves.

os.makedirs("models", exist_ok=True)

early_stop = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

checkpoint = callbacks.ModelCheckpoint(
    filepath=MODEL_OUT,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# ── 8. Train ──────────────────────────────────────────────────────────────────

print("\nStarting training…")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,          # <-- fixes the broken early stopping bug
    class_weight=class_weight_dict,  # <-- handles trash imbalance
    callbacks=[early_stop, checkpoint]
)

# ── 9. Phase 2 — Fine-tuning ──────────────────────────────────────────────────
# Unfreeze the top layers of EfficientNetB0 so they can adapt to waste images.
# Use a learning rate 10× smaller than phase 1 so we nudge, not overwrite.
# EfficientNetB0 has 237 layers — unfreeze from layer 200 onwards (top ~37 layers).

FINE_TUNE_FROM = 200          # unfreeze layers from index 200 onwards
MODEL_OUT_FT   = "models/efficientnetb0_finetuned.keras"  # checkpoint during phase 2

print("\n── Phase 2: Fine-tuning ──")
base_model.trainable = True

# Re-freeze everything below FINE_TUNE_FROM.
# BatchNormalization layers are ALWAYS kept frozen — unfreezing BN with a small
# dataset corrupts the running mean/variance statistics and hurts accuracy.
for layer in base_model.layers[:FINE_TUNE_FROM]:
    layer.trainable = False
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False

trainable_count = sum(1 for l in base_model.layers if l.trainable)
print(f"Unfroze {trainable_count} layers in EfficientNetB0 base (from layer {FINE_TUNE_FROM}, BN frozen)")

# Recompile with a much lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-4),   # 10× smaller than phase 1
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_ft = callbacks.ModelCheckpoint(
    filepath=MODEL_OUT_FT,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

early_stop_ft = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("Starting fine-tuning…")
history_ft = model.fit(
    train_ds,
    epochs=30,                           # shorter — we're just nudging
    validation_data=val_ds,
    class_weight=class_weight_dict,
    callbacks=[early_stop_ft, checkpoint_ft]
)

# ── 10. Final test evaluation ─────────────────────────────────────────────────

print("\nEvaluating on held-out test set…")
test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# Export as SavedModel format (folder) — required for TFLite conversion
# Note: model.export() is the correct method in Keras 3+, not model.save()
SAVED_MODEL_PATH = "models/efficientnetb0_savedmodel"
model.export(SAVED_MODEL_PATH)
print(f"\nFine-tuned model exported to: {SAVED_MODEL_PATH}/")
