#!/usr/bin/env python3
"""
DataPrep_v3_2.py

Final high-quality preprocessing pipeline for volumetric CT classification.

Features:
 - Loads DICOM series → HU
 - Computes lung mask (background removal)
 - Applies 3 CT windows (lung, soft, bone)
 - Builds volume (Z,H,W,3)
 - Resamples using full 3D cubic interpolation (scipy.ndimage.zoom)
 - Saves uniform .npy volumes
 - Generates train/val/test splits (stratified by label)
 - Copies volumes into: train/, val/, test/ directories

Usage:
    python DataPrep_v3_2.py
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import zoom
from scipy.ndimage import binary_fill_holes
from skimage.morphology import closing, opening, disk, remove_small_objects
from sklearn.model_selection import train_test_split
import shutil

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "Data"
OUTPUT_ROOT = "preprocessed_v3_1"     # keep folder name
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABEL_MAP = {
    "normal": "normal",
    "covid-19": "covid",
    "cap": "cap"
}

# Target shape for cubic interpolation
TARGET_Z = 48
TARGET_H = 64
TARGET_W = 64

# Train/Val/Test ratio
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
RANDOM_SEED = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_hu_volume(study_path):
    """Load full DICOM series → HU array."""
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(study_path)

    if len(dcm_files) == 0:
        raise RuntimeError(f"No DICOM files found in {study_path}")

    reader.SetFileNames(dcm_files)
    img = reader.Execute()

    arr = sitk.GetArrayFromImage(img).astype(np.int16)

    slope = float(img.GetMetaData("0028|1053")) if img.HasMetaDataKey("0028|1053") else 1
    intercept = float(img.GetMetaData("0028|1052")) if img.HasMetaDataKey("0028|1052") else 0
    arr = arr * slope + intercept

    return arr


def apply_slicewise(volume, func):
    out = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        out[i] = func(volume[i])
    return out


def get_lung_mask(hu):
    """ Basic lung segmentation to remove ribs/heart/background. """
    mask = (hu < -400)

    mask = apply_slicewise(mask, lambda x: opening(x, disk(3)))
    mask = apply_slicewise(mask, lambda x: closing(x, disk(5)))

    cleaned = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        cleaned[i] = remove_small_objects(mask[i].astype(bool), min_size=500)

    filled = np.zeros_like(cleaned)
    for i in range(cleaned.shape[0]):
        filled[i] = binary_fill_holes(cleaned[i])

    return filled.astype(np.uint8)


def window(hu, level, width):
    lower = level - width//2
    upper = level + width//2
    w = np.clip(hu, lower, upper)
    return (w - lower) / (upper - lower)


def resample_volume_3d(vol, target_z, target_h, target_w):
    """3D cubic interpolation to fixed shape."""
    Z, H, W, C = vol.shape

    zoom_factors = (
        target_z / Z,
        target_h / H,
        target_w / W
    )

    out = np.zeros((target_z, target_h, target_w, C), dtype=np.float32)

    for c in range(C):
        out[..., c] = zoom(vol[..., c], zoom_factors, order=3)

    return out


# ============================================================
# PIPELINE
# ============================================================

records = []

for folder in os.listdir(DATA_ROOT):
    folder_path = os.path.join(DATA_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    csv_path = os.path.join(DATA_ROOT, f"{folder.lower()}-labels.csv")
    if not os.path.exists(csv_path):
        print(f"[WARN] Missing CSV for {folder}")
        continue

    df = pd.read_csv(csv_path)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {folder}"):

        study = row["Study"]
        label = LABEL_MAP[row["Label"]]
        study_path = os.path.join(DATA_ROOT, folder, study)

        if not os.path.exists(study_path):
            print(f"[WARN] Missing folder: {study_path}")
            continue

        try:
            # 1. Load volume
            hu = load_hu_volume(study_path)

            # 2. Lung mask
            lung = get_lung_mask(hu)
            hu = hu * lung

            # -------------------------
            # 3. Create 4-channel volume
            # -------------------------

            # Channel 0: raw HU values inside lung
            raw_hu = np.clip(hu, -1024, 400).astype(np.float32)
            raw_hu = (raw_hu + 1024) / (400 + 1024)   # normalize to [0,1]

            # Channel 1: lung window
            lung_win = window(hu, -600, 1500)  # already 0..1

            # Channel 2: soft-tissue window
            soft_win = window(hu, 40, 400)     # 0..1

            # Channel 3: bone window
            bone_win = window(hu, 300, 1500)   # 0..1

            # Stack → (Z, H, W, 4)
            vol = np.stack([raw_hu, lung_win, soft_win, bone_win], axis=-1)
            vol = resample_volume_3d(vol, TARGET_Z, TARGET_H, TARGET_W)

            # 5. Save
            out_dir = os.path.join(OUTPUT_ROOT, label)
            os.makedirs(out_dir, exist_ok=True)

            out_path = os.path.join(out_dir, f"{study}.npy")
            np.save(out_path, vol.astype(np.float32))

            records.append([out_path, label, study])

        except Exception as e:
            print(f"[ERROR] {study}: {e}")
            continue


# ============================================================
# SPLIT INTO TRAIN / VAL / TEST
# ============================================================

df_all = pd.DataFrame(records, columns=["npy_path", "label", "study"])
df_all = df_all[df_all["npy_path"].apply(os.path.exists)].drop_duplicates(subset=["study"])

print("\n[INFO] Total processed studies:", len(df_all))

# First split into train + temp (val+test)
train_df, temp_df = train_test_split(
    df_all,
    test_size=(1 - TRAIN_RATIO),
    stratify=df_all["label"],
    random_state=RANDOM_SEED
)

# Split temp → val + test
val_frac = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
val_df, test_df = train_test_split(
    temp_df,
    test_size=(1 - val_frac),
    stratify=temp_df["label"],
    random_state=RANDOM_SEED
)

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

df_split = pd.concat([train_df, val_df, test_df], ignore_index=True)


# ============================================================
# COPY FILES INTO SPLIT FOLDERS
# ============================================================

def copy_into_split(df, split_name):
    for _, r in df.iterrows():
        src = r["npy_path"]
        label = r["label"]
        fname = os.path.basename(src)

        dst_dir = os.path.join(OUTPUT_ROOT, split_name, label)
        os.makedirs(dst_dir, exist_ok=True)

        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)


copy_into_split(train_df, "train")
copy_into_split(val_df,   "val")
copy_into_split(test_df,  "test")


# Save final index
index_path = os.path.join(OUTPUT_ROOT, "index_with_splits.csv")
df_split.to_csv(index_path, index=False)

print("\n[✓] Split complete!")
print(df_split["split"].value_counts())
print(f"Index saved to: {index_path}")
print(f"Files copied to {OUTPUT_ROOT}/train, val, test/")
