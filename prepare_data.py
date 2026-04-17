"""
prepare_data.py — Merge TrashNet + Garbage Dataset into one training directory

Input:
  Data/raw/          — TrashNet (6 classes, 2,527 images)
  Data/original/     — Garbage Dataset GD (10 classes, 12,259 images)

Output:
  Data/merged/       — 9 classes, ~14,786 images
    cardboard/       — TrashNet + GD
    glass/           — TrashNet + GD
    metal/           — TrashNet + GD
    paper/           — TrashNet + GD
    plastic/         — TrashNet + GD
    trash/           — TrashNet + GD
    biological/      — GD only (food waste)
    battery/         — GD only (hazardous)
    textiles/        — GD shoes + clothes merged

Note: shoes and clothes are merged into one "textiles" class because
both are depot/donation items — splitting them adds no user value.

Run in Colab:
  !python prepare_data.py
"""

import os
import shutil

# ── Config ────────────────────────────────────────────────────────────────────

TRASHNET_DIR = "data/raw"   # lowercase — TrashNet folder on Drive
GD_DIR       = "Data/original"
MERGED_DIR   = "Data/merged"

# Classes that exist in both datasets — merge from both sources
SHARED_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Classes that only exist in GD — copy directly
GD_ONLY_CLASSES = ["biological", "battery"]

# GD classes to merge into a single "textiles" class
TEXTILES_CLASSES = ["shoes", "clothes"]

# ── Helper ────────────────────────────────────────────────────────────────────

def copy_images(src_dir, dst_dir, prefix):
    """
    Copy all images from src_dir into dst_dir.
    Adds a prefix to each filename to avoid collisions when merging
    two sources into the same folder.
    """
    os.makedirs(dst_dir, exist_ok=True)
    copied = 0
    for fname in os.listdir(src_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            src = os.path.join(src_dir, fname)
            dst = os.path.join(dst_dir, f"{prefix}_{fname}")
            shutil.copy2(src, dst)
            copied += 1
    return copied

# ── Main ──────────────────────────────────────────────────────────────────────

print("Building merged dataset…\n")
total = 0

# 1. Shared classes — copy from both TrashNet and GD
for cls in SHARED_CLASSES:
    dst = os.path.join(MERGED_DIR, cls)

    trashnet_src = os.path.join(TRASHNET_DIR, cls)
    gd_src       = os.path.join(GD_DIR, cls)

    n1 = copy_images(trashnet_src, dst, prefix="tn") if os.path.exists(trashnet_src) else 0
    n2 = copy_images(gd_src,       dst, prefix="gd") if os.path.exists(gd_src)       else 0

    print(f"{cls:12s}  TrashNet: {n1:4d}  GD: {n2:4d}  Total: {n1+n2:4d}")
    total += n1 + n2

# 2. GD-only classes — copy directly
for cls in GD_ONLY_CLASSES:
    dst    = os.path.join(MERGED_DIR, cls)
    gd_src = os.path.join(GD_DIR, cls)

    n = copy_images(gd_src, dst, prefix="gd") if os.path.exists(gd_src) else 0
    print(f"{cls:12s}  GD only:  {n:4d}  Total: {n:4d}")
    total += n

# 3. Textiles — merge shoes + clothes into one folder
dst = os.path.join(MERGED_DIR, "textiles")
n_textiles = 0
for cls in TEXTILES_CLASSES:
    gd_src = os.path.join(GD_DIR, cls)
    n = copy_images(gd_src, dst, prefix=f"gd_{cls}") if os.path.exists(gd_src) else 0
    n_textiles += n
print(f"{'textiles':12s}  GD only:  {n_textiles:4d}  Total: {n_textiles:4d}")
total += n_textiles

# ── Summary ───────────────────────────────────────────────────────────────────

print(f"\n{'─'*45}")
print(f"Total images in merged dataset: {total:,}")
print(f"Saved to: {MERGED_DIR}/")
print("\nClass counts in merged dataset:")
for cls in sorted(os.listdir(MERGED_DIR)):
    cls_path = os.path.join(MERGED_DIR, cls)
    if os.path.isdir(cls_path):
        count = len([f for f in os.listdir(cls_path)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  {cls:12s}: {count:,}")
