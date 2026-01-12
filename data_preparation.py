import os
import shutil
import random

# ================= CONFIG =================
SOURCE_DIR = "data"            # folder with images + labels together
OUTPUT_DIR = "dataset_split"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

IMAGE_EXTENSIONS = (".jpg", ".png", ".jpeg")
SEED = 42
# =========================================

random.seed(SEED)

# Create output directories
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Collect image files
image_files = [
    f for f in os.listdir(SOURCE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

random.shuffle(image_files)

total = len(image_files)
train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_files = image_files[:train_end]
val_files = image_files[train_end:val_end]
test_files = image_files[val_end:]

def copy_pair(img_name, split):
    img_src = os.path.join(SOURCE_DIR, img_name)
    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_src = os.path.join(SOURCE_DIR, label_name)

    img_dst = os.path.join(OUTPUT_DIR, split, "images", img_name)
    label_dst = os.path.join(OUTPUT_DIR, split, "labels", label_name)

    shutil.copy(img_src, img_dst)

    if os.path.exists(label_src):
        shutil.copy(label_src, label_dst)
    else:
        print(f"⚠ Warning: Missing label for {img_name}")

# Copy data
for img in train_files:
    copy_pair(img, "train")

for img in val_files:
    copy_pair(img, "val")

for img in test_files:
    copy_pair(img, "test")

print("✅ Dataset split completed successfully!")
print(f"Train: {len(train_files)} images")
print(f"Val  : {len(val_files)} images")
print(f"Test : {len(test_files)} images")
