#!/usr/bin/env python3
"""
train_small3d_tf.py

Ultra-small 3D CNN for training on (Z,H,W,3) preprocessed mask volumes.
No preprocessing is repeated here — only resize + depth crop/pad.

Default:
    slices = 48
    spatial = 48
    params ~ 70k total

Usage:
    python train_small3d_tf.py --index preprocessed_v2/preprocessed_index.csv --out runs/small_run1
"""

import os
import argparse
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from skimage.transform import resize
from sklearn.preprocessing import label_binarize

# -----------------------------
# CONFIG
# -----------------------------
DEFAULT_SLICES = 48
DEFAULT_SPATIAL = 48      # MUCH smaller → faster + less overfit
DEFAULT_BATCH = 2
DEFAULT_EPOCHS = 20
SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# -----------------------------
# Utility Functions
# -----------------------------
def crop_or_pad_depth(vol, target_slices):
    z, h, w, c = vol.shape
    if z == target_slices:
        return vol
    if z > target_slices:
        start = (z - target_slices) // 2
        return vol[start:start + target_slices]

    pad = target_slices - z
    pad_front = pad // 2
    pad_back  = pad - pad_front

    before = np.repeat(vol[[0]], pad_front, axis=0) if pad_front > 0 else np.zeros((0,h,w,c))
    after  = np.repeat(vol[[-1]], pad_back, axis=0) if pad_back > 0 else np.zeros((0,h,w,c))

    return np.concatenate([before, vol, after], axis=0)


def resize_volume(vol, target_h, target_w):
    Z,H,W,C = vol.shape
    out = np.zeros((Z, target_h, target_w, C), dtype=np.float32)
    for i in range(Z):
        sl = vol[i].astype(np.float32)
        out[i] = np.stack([
            resize(sl[..., ch], (target_h, target_w), order=1, preserve_range=True, anti_aliasing=False)
            for ch in range(C)
        ], axis=-1)
    return out


def augment(vol):
    # VERY LIGHT AUGMENTATION (helps generalization)
    if random.random() < 0.5:
        vol = np.flip(vol, axis=2)
    if random.random() < 0.3:
        vol = np.flip(vol, axis=1)
    if random.random() < 0.25:
        k = random.randint(1,3)
        vol = np.rot90(vol, k=k, axes=(1,2))
    return vol


# -----------------------------
# Dataset Pipeline (ONLY resizing + small augment)
# -----------------------------
def npy_generator(df, slices, H, W, label_map, augment_flag):
    for _, row in df.iterrows():
        path  = row["npy_path"]
        label = row["label"]

        try:
            vol = np.load(path).astype(np.float32)  # (Z,H,W,3)
        except:
            continue

        # Resize ONLY
        vol = crop_or_pad_depth(vol, slices)
        vol = resize_volume(vol, H, W)

        # Normalize to [0,1]
        vol = np.clip(vol, 0, 1)

        if augment_flag:
            vol = augment(vol)

        yield vol, np.int32(label_map[label])


def build_dataset(df, batch, slices, H, W, label_map, augment_flag=False, shuffle=False):
    gen = lambda: npy_generator(df, slices, H, W, label_map, augment_flag)

    output_types  = (tf.float32, tf.int32)
    output_shapes = ((slices, H, W, 3), ())

    ds = tf.data.Dataset.from_generator(gen, output_types, output_shapes)

    if shuffle:
        ds = ds.shuffle(64, seed=SEED)

    ds = ds.batch(batch)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# -----------------------------
# ULTRA SMALL 3D CNN
# (~70k params)
# -----------------------------
def build_tiny3d(input_shape, num_classes):
    inp = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv3D(16, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool3D(2)(x)

    x = tf.keras.layers.Conv3D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool3D(2)(x)

    x = tf.keras.layers.Conv3D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    out = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    return tf.keras.Model(inp, out)


# -----------------------------
# Main Train Loop
# -----------------------------
def main(args):
    df = pd.read_csv(args.index)

    train_df = df[df["split"]=="train"].reset_index(drop=True)
    val_df   = df[df["split"]=="val"].reset_index(drop=True)
    test_df  = df[df["split"]=="test"].reset_index(drop=True)

    print(f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    labels = sorted(df["label"].unique().tolist())
    label_map = {lab: i for i, lab in enumerate(labels)}
    print("Label map:", label_map)

    train_ds = build_dataset(train_df, args.batch_size, args.slices, args.spatial, args.spatial,
                             label_map, augment_flag=True, shuffle=True)
    val_ds   = build_dataset(val_df, args.batch_size, args.slices, args.spatial, args.spatial,
                             label_map, augment_flag=False, shuffle=False)
    test_ds  = build_dataset(test_df, args.batch_size, args.slices, args.spatial, args.spatial,
                             label_map, augment_flag=False, shuffle=False)

    num_classes = len(labels)
    input_shape = (args.slices, args.spatial, args.spatial, 3)

    model = build_tiny3d(input_shape, num_classes)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    os.makedirs(args.out, exist_ok=True)

    ckpt_path = os.path.join(args.out, "best_model.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        monitor="val_sparse_categorical_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=5,
        mode="max",
        restore_best_weights=True
    )

    tb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.out, "tensorboard"))

    # TRAIN
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, early, tb]
    )

    final_path = os.path.join(args.out, "final_model.h5")
    model.save(final_path)
    print("Saved final model to:", final_path)

    # -------------------------
    # Evaluate on Test
    # -------------------------
    y_true = []
    y_prob = []

    for xb, yb in tqdm(test_ds, desc="Testing"):
        pred = model.predict(xb)
        y_prob.extend(pred.tolist())
        y_true.extend(yb.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Per-class AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        aucs = {}
        for i, lab in enumerate(labels):
            try:
                auc_val = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            except:
                auc_val = float("nan")
            aucs[lab] = auc_val
        print("\nPer-class AUC:", aucs)
    except:
        print("AUC computation skipped.")

    # Save history
    import json
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump(history.history, f, indent=2)

    print("\nDONE. Run outputs saved to:", args.out)


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--out", default="runs/small_run1")
    parser.add_argument("--slices", type=int, default=DEFAULT_SLICES)
    parser.add_argument("--spatial", type=int, default=DEFAULT_SPATIAL)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    main(args)


