import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import SimpleITK as sitk  # ← NEW

# --------------------
# CONFIG
# --------------------
DATA_ROOT = "Data"                 # your root data folder
OUTPUT_ROOT = "Processed"          # output folder for PNGs
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# LABEL NORMALIZATION
LABEL_MAP = {
    "normal": "normal",
    "covid-19": "covid",
    "cap": "cap"
}

# MASTER CSV
records = []

def normalize(img):
    img = img.astype(np.float32)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    return img


# ------------------------------------------------------------
# PROCESS EVERY Test-{i} FOLDER
# ------------------------------------------------------------
for test_folder in os.listdir(DATA_ROOT):
    
    folder_path = os.path.join(DATA_ROOT, test_folder)
    if os.path.isdir(folder_path):
        print(f"[DEBUG] Current folder: {test_folder.lower()}, Looking for CSV: {test_folder.lower()}-labels.csv")
        if not test_folder.lower().startswith("test-"):
            continue

        # CSV path is always in DATA_ROOT, lowercase
        csv_path = os.path.join(DATA_ROOT, f"{test_folder.lower()}-labels.csv")

        if not os.path.exists(csv_path):
            print(f"[WARN] Missing CSV for: {test_folder}")
            continue

        print(f"\n[INFO] Processing {test_folder} …")
        df = pd.read_csv(csv_path)

        # Dict: "T2-001" → "covid-19"
        study_to_label = dict(zip(df["Study"], df["Label"]))

        # ------------------------------------------------------------
        # PROCESS EACH STUDY FOLDER
        # ------------------------------------------------------------
        study_folders = [
            f for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]

        for study_name in tqdm(study_folders, desc=f"{test_folder} studies"):
            study_path = os.path.join(folder_path, study_name)

            if study_name not in study_to_label:
                continue

            label = LABEL_MAP[study_to_label[study_name]]

            # Create output folder: Processed/<label>/<study>/
            output_study_path = os.path.join(OUTPUT_ROOT, label, study_name)
            os.makedirs(output_study_path, exist_ok=True)

            # ------------------------------------------------------------
            # READ THE ENTIRE STUDY USING SimpleITK
            # ------------------------------------------------------------
            reader = sitk.ImageSeriesReader()

            try:
                dicom_names = reader.GetGDCMSeriesFileNames(study_path)
                reader.SetFileNames(dicom_names)
                image = reader.Execute()
                volume = sitk.GetArrayFromImage(image)  # shape: (Z, H, W)
            except Exception as e:
                print(f"[ERROR] Failed reading {study_name}: {e}")
                continue

            # ------------------------------------------------------------
            # Convert every slice in the volume
            # ------------------------------------------------------------
            for i, slice_arr in enumerate(tqdm(volume, leave=False, desc=f"{study_name} slices")):
                img = normalize(slice_arr)

                png_path = os.path.join(output_study_path, f"slice_{i:04d}.png")
                Image.fromarray(img).save(png_path)

                records.append([png_path, label, study_name, test_folder])


# ------------------------------------------------------------
# SAVE MASTER CSV
# ------------------------------------------------------------
df_out = pd.DataFrame(records, columns=["png_path", "label", "study", "test_set"])
df_out.to_csv(os.path.join(OUTPUT_ROOT, "all_images_with_labels.csv"), index=False)

print("\n[✓] Conversion Complete — PNGs + CSV generated successfully.")
print(f"[✓] Output written to: {OUTPUT_ROOT}/")
