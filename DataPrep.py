import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from skimage.morphology import closing, opening, disk, remove_small_objects
from scipy.ndimage import binary_fill_holes
import random

# =====================
# CONFIG
# =====================

DATA_ROOT = "Data"
OUTPUT_ROOT = "preprocessed_v2"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

LABEL_MAP = {
    "normal": "normal",
    "covid-19": "covid",
    "cap": "cap"
}

TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15


# =====================
# HELPERS
# =====================

def apply_slicewise(volume, func):
    """Apply 2D morphology slice-by-slice to a 3D volume."""
    out = np.zeros_like(volume)
    for i in range(volume.shape[0]):
        out[i] = func(volume[i])
    return out


def load_hu_volume(study_path):
    """Load DICOM study folder → return HU volume."""
    reader = sitk.ImageSeriesReader()
    dcm_files = reader.GetGDCMSeriesFileNames(study_path)
    reader.SetFileNames(dcm_files)

    image = reader.Execute()
    vol = sitk.GetArrayFromImage(image).astype(np.int16)  # (Z,H,W)

    # HU conversion
    slope = float(image.GetMetaData("0028|1053")) if image.HasMetaDataKey("0028|1053") else 1
    intercept = float(image.GetMetaData("0028|1052")) if image.HasMetaDataKey("0028|1052") else 0

    hu = vol * slope + intercept
    return hu


# =====================
# FEATURE MASKS
# =====================

def get_lung_mask(hu):
    """Lung segmentation mask using HU threshold + morphology."""
    lung = (hu < -400)  # binary mask

    # 2D morphology slice-wise
    lung = apply_slicewise(lung, lambda x: opening(x, disk(3)))
    lung = apply_slicewise(lung, lambda x: closing(x, disk(5)))

    # Remove small junk slice-wise
    cleaned = np.zeros_like(lung)
    for i in range(lung.shape[0]):
        cleaned[i] = remove_small_objects(lung[i].astype(bool), min_size=500)

    # Fill holes slice-wise
    filled = np.zeros_like(cleaned)
    for i in range(cleaned.shape[0]):
        filled[i] = binary_fill_holes(cleaned[i])

    return filled.astype(np.uint8)


def get_ggo_mask(hu):
    """Ground-glass opacity mask (COVID indicator)."""
    mask = (hu > -750) & (hu < -300)

    mask = apply_slicewise(mask, lambda x: closing(x, disk(3)))
    mask = apply_slicewise(mask, lambda x: opening(x, disk(2)))

    return mask.astype(np.uint8)


def get_consolidation_mask(hu):
    """Dense consolidation mask (CAP indicator)."""
    mask = (hu > -300) & (hu < 150)

    mask = apply_slicewise(mask, lambda x: closing(x, disk(5)))

    cleaned = np.zeros_like(mask)
    for i in range(mask.shape[0]):
        cleaned[i] = remove_small_objects(mask[i].astype(bool), min_size=1500)

    filled = np.zeros_like(cleaned)
    for i in range(cleaned.shape[0]):
        filled[i] = binary_fill_holes(cleaned[i])

    return filled.astype(np.uint8)


def save_volume(vol, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, vol)


# ===============================
# COLLECT ALL STUDIES
# ===============================

all_studies = []

for folder in os.listdir(DATA_ROOT):
    if not folder.lower().startswith("test-"):
        continue

    csv_path = os.path.join(DATA_ROOT, f"{folder.lower()}-labels.csv")
    if not os.path.exists(csv_path):
        continue

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        study = row["Study"]
        label = LABEL_MAP[row["Label"]]
        study_path = os.path.join(DATA_ROOT, folder, study)

        if os.path.exists(study_path):
            all_studies.append((study_path, study, label))


# ===============================
# SPLIT: Train / Val / Test
# ===============================

random.shuffle(all_studies)
N = len(all_studies)
train_n = int(TRAIN_SPLIT * N)
val_n = int(VAL_SPLIT * N)

train_set = all_studies[:train_n]
val_set = all_studies[train_n:train_n+val_n]
test_set = all_studies[train_n+val_n:]

splits = [
    ("train", train_set),
    ("val", val_set),
    ("test", test_set)
]


# ===============================
# PROCESS VOLUMES
# ===============================

records = []

for split_name, split_data in splits:
    print(f"\n[INFO] Processing {split_name.upper()} — {len(split_data)} studies")

    for study_path, study_name, label in tqdm(split_data):
        # 1. Load HU
        hu = load_hu_volume(study_path)

        # 2. Masks
        lung = get_lung_mask(hu)
        ggo = get_ggo_mask(hu) * lung
        cons = get_consolidation_mask(hu) * lung

        # 3. Stack channels
        volume = np.stack([ggo, cons, lung], axis=-1).astype(np.uint8)

        # 4. Save volume
        out_path = os.path.join(OUTPUT_ROOT, split_name, label, f"{study_name}.npy")
        save_volume(volume, out_path)

        records.append([out_path, label, study_name, split_name])


# ===============================
# SAVE DATA INDEX
# ===============================

df_out = pd.DataFrame(records, columns=["npy_path", "label", "study", "split"])
df_out.to_csv(os.path.join(OUTPUT_ROOT, "preprocessed_index.csv"), index=False)

print("\n[✓] DONE — Preprocessing complete.")
