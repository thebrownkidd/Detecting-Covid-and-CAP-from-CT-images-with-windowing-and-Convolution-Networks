#!/usr/bin/env python3
"""
DataPrep_v3_3_multithread.py

Fast multithreaded preprocessing:
 - Loads DICOM → HU
 - Lung mask
 - CT windows (raw, lung, soft, bone)
 - 4-channel volume
 - Cubic interpolation (3D)
 - Saves .npy
 - Writes single index CSV (no splits)

Uses ThreadPoolExecutor to max out your AMD CPU.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import zoom, binary_fill_holes
from skimage.morphology import opening, closing, remove_small_objects, disk
from concurrent.futures import ThreadPoolExecutor, as_completed

# --------------------
# CONFIG
# --------------------
DATA_ROOT = "Data"
OUTPUT_ROOT = "preprocessed_v3_1"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABEL_MAP = {
    "normal": "normal",
    "covid-19": "covid",
    "cap": "cap"
}

TARGET_Z = 48
TARGET_H = 64
TARGET_W = 64

MAX_WORKERS = min(os.cpu_count(), 8)  # Use up to 8 CPU threads
print(f"[i] Using {MAX_WORKERS} threads for preprocessing.")

# --------------------
# Helper functions
# --------------------

def load_hu(study_path):
    """Load DICOM → HU"""
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(study_path)
    if not dcm_files:
        raise RuntimeError(f"No DICOM series in {study_path}")

    reader.SetFileNames(dcm_files)
    img = reader.Execute()

    arr = sitk.GetArrayFromImage(img).astype(np.int16)

    slope = float(img.GetMetaData("0028|1053")) if img.HasMetaDataKey("0028|1053") else 1
    intercept = float(img.GetMetaData("0028|1052")) if img.HasMetaDataKey("0028|1052") else 0

    return arr * slope + intercept


def slice_apply(vol, func):
    out = np.zeros_like(vol)
    for i in range(vol.shape[0]):
        out[i] = func(vol[i])
    return out


def lung_mask(hu):
    mask = (hu < -400)
    mask = slice_apply(mask, lambda x: opening(x, disk(3)))
    mask = slice_apply(mask, lambda x: closing(x, disk(5)))

    cleaned = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        cleaned[i] = remove_small_objects(mask[i].astype(bool), min_size=500)

    filled = np.zeros_like(cleaned)
    for i in range(cleaned.shape[0]):
        filled[i] = binary_fill_holes(cleaned[i])

    return filled.astype(np.uint8)


def window(hu, level, width):
    lo = level - width // 2
    hi = level + width // 2
    w = np.clip(hu, lo, hi)
    return (w - lo) / (hi - lo)


def resample_3d(vol):
    """3D cubic interpolation to fixed shape"""
    Z, H, W, C = vol.shape
    zoom_factors = (TARGET_Z / Z, TARGET_H / H, TARGET_W / W)
    out = np.zeros((TARGET_Z, TARGET_H, TARGET_W, C), dtype=np.float32)
    for c in range(C):
        out[..., c] = zoom(vol[..., c], zoom_factors, order=3)
    return out


# --------------------
# Worker function
# --------------------

def process_study(study_path, label, study_name):
    """
    Runs full preprocessing for ONE study.
    Returns: (output_path, label, study_name)
    """
    try:
        # Load HU
        hu = load_hu(study_path)

        # Lung segmentation
        mask = lung_mask(hu)
        hu = hu * mask

        # Channels
        raw = np.clip(hu, -1024, 400).astype(np.float32)
        raw = (raw + 1024) / (400 + 1024)

        lung_win = window(hu, -600, 1500)
        soft_win = window(hu, 40, 400)
        bone_win = window(hu, 300, 1500)

        vol = np.stack([raw, lung_win, soft_win, bone_win], axis=-1)
        vol = resample_3d(vol)

        # Save
        out_dir = os.path.join(OUTPUT_ROOT, label)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{study_name}.npy")

        np.save(out_path, vol.astype(np.float32))
        return out_path, label, study_name

    except Exception as e:
        return None, None, f"Error in {study_path}: {e}"


# --------------------
# Main loop
# --------------------

records = []
tasks = []

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    for folder in os.listdir(DATA_ROOT):

        csv_path = os.path.join(DATA_ROOT, f"{folder.lower()}-labels.csv")
        folder_path = os.path.join(DATA_ROOT, folder)

        if not os.path.isdir(folder_path) or not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            study = row["Study"]
            label = LABEL_MAP[row["Label"]]
            study_path = os.path.join(folder_path, study)

            tasks.append(
                executor.submit(process_study, study_path, label, study)
            )

    # Collect results with progress bar
    for fut in tqdm(as_completed(tasks), total=len(tasks), desc="Processing studies"):
        out_path, label, study = fut.result()
        if out_path is not None:
            records.append([out_path, label, study])


# Save final index CSV
df_out = pd.DataFrame(records, columns=["npy_path", "label", "study"])
df_out.to_csv(os.path.join(OUTPUT_ROOT, "index.csv"), index=False)

print("\n[✓] Multithreaded preprocessing complete!")
print("Total studies:", len(df_out))
print("Index saved to:", os.path.join(OUTPUT_ROOT, "index.csv"))
