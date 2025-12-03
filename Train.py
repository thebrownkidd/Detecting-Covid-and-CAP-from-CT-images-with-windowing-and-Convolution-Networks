#!/usr/bin/env python3
"""
train_nano3d_v3.py

Train a nano 3D CNN on preprocessed_v3_1 volumes.
Assumes index CSV has columns: npy_path,label,study,split

Usage:
    python train_nano3d_v3.py --index preprocessed_v3_1/index_with_splits.csv --out runs/nano_v3
"""

import os
import argparse
import numpy as np
import pandas as pd
import random
import json
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# -------------------------
# Data pipeline functions
# -------------------------
def npy_loader(path):
    """Load a .npy volume and ensure dtype float32 and expected shape."""
    vol = np.load(path)
    # ensure float32 and clipped to [0,1] (preproc should have done this)
    vol = vol.astype(np.float32)
    if vol.max() > 1.5:  # quick sanity check
        vol = np.clip(vol, 0.0, 1.0)
    return vol

def generator_from_df(df, label_map, augment=False):
    """
    Yields (vol, label_idx).
    df must contain column 'npy_path' and 'label'
    """
    for _, row in df.iterrows():
        try:
            vol = np.load(row['npy_path']).astype(np.float32)  # (Z,H,W,C)
        except Exception as e:
            print(f"[WARN] failed loading {row['npy_path']}: {e}")
            continue

        # basic sanity: ensure channel last 4D
        if vol.ndim == 3:
            vol = vol[..., None]

        # light augmentation (in numpy)
        if augment:
            # flip LR
            if random.random() < 0.5:
                vol = np.flip(vol, axis=2)
            # flip UD
            if random.random() < 0.3:
                vol = np.flip(vol, axis=1)
            # rotate 90 deg occasionally
            if random.random() < 0.2:
                k = random.randint(1,3)
                vol = np.rot90(vol, k=k, axes=(1,2))

        yield vol, np.int32(label_map[row['label']])


def build_tf_dataset(df, label_map, batch, augment=False, shuffle=False):
    gen = lambda: generator_from_df(df, label_map, augment=augment)
    # infer shape from first sample
    for _, r in df.iterrows():
        try:
            sample = np.load(r['npy_path'])
            break
        except:
            sample = None
    if sample is None:
        raise RuntimeError("No readable .npy found in dataframe.")
    # shape handling
    if sample.ndim == 3:
        sample_shape = (sample.shape[0], sample.shape[1], sample.shape[2], 1)
    else:
        sample_shape = (sample.shape[0], sample.shape[1], sample.shape[2], sample.shape[3])

    output_types = (tf.float32, tf.int32)
    output_shapes = (sample_shape, ())

    ds = tf.data.Dataset.from_generator(gen, output_types=output_types, output_shapes=output_shapes)
    if shuffle:
        ds = ds.shuffle(buffer_size=64, seed=SEED)
    ds = ds.batch(batch)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# -------------------------
# Nano 3D model (~18k params)
# -------------------------
def conv3d_bn_relu(x, filters, kernel=3):
    x = tf.keras.layers.Conv3D(filters, kernel, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return tf.keras.layers.ReLU()(x)

def residual_block(x, filters):
    shortcut = x
    x = conv3d_bn_relu(x, filters)
    x = tf.keras.layers.Conv3D(filters, 3, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    # residual add (handles channel mismatch)
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv3D(filters, 1, padding="same", use_bias=False)(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)

    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    return x

def build_nano3d(input_shape, n_classes):
    inp = tf.keras.Input(shape=input_shape)

    # Block 1 — small but expressive
    x = conv3d_bn_relu(inp, 16)
    x = residual_block(x, 16)
    x = tf.keras.layers.MaxPool3D(2)(x)  # Z/2, H/2, W/2

    # Block 2 — more channels
    x = conv3d_bn_relu(x, 32)
    x = residual_block(x, 32)
    x = tf.keras.layers.MaxPool3D(2)(x)

    # Block 3 — deeper features
    x = conv3d_bn_relu(x, 48)
    x = residual_block(x, 48)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    # Dense head
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    out = tf.keras.layers.Dense(n_classes, activation="softmax")(x)

    model = tf.keras.Model(inp, out)
    return model


# -------------------------
# Training + evaluation
# -------------------------
def main(args):
    # load index
    df = pd.read_csv(args.index)
    required_cols = {'npy_path','label','split'}
    if not required_cols.issubset(set(df.columns)):
        raise RuntimeError(f"Index CSV must contain columns: {required_cols}")

    # remove missing
    df = df[df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError("No .npy files found per index.")

    # build label map consistent across splits
    labels_sorted = sorted(df['label'].unique().tolist())
    label_map = {lab:i for i,lab in enumerate(labels_sorted)}
    print("Labels:", labels_sorted)
    print("Label map:", label_map)

    train_df = df[df['split']=='train'].reset_index(drop=True)
    val_df   = df[df['split']=='val'].reset_index(drop=True)
    test_df  = df[df['split']=='test'].reset_index(drop=True)
    print(f"Samples: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # compute class weights from train only
    class_weights = None
    if args.use_class_weights:
        y = train_df['label'].map(label_map).values
        cw = compute_class_weight(class_weight='balanced', classes=np.arange(len(labels_sorted)), y=y)
        class_weights = {i: float(w) for i,w in enumerate(cw)}
        print("Using class weights:", class_weights)

    # build tf datasets
    train_ds = build_tf_dataset(train_df, label_map, batch=args.batch_size, augment=True, shuffle=True)
    val_ds   = build_tf_dataset(val_df,   label_map, batch=args.batch_size, augment=False, shuffle=False)
    test_ds  = build_tf_dataset(test_df,  label_map, batch=args.batch_size, augment=False, shuffle=False)

    # infer input shape from dataset element
    for xb, yb in train_ds.take(1):
        input_shape = xb.shape[1:]  # (Z,H,W,C)
        break

    model = build_nano3d(input_shape, n_classes=len(labels_sorted))
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_categorical_accuracy')]
    )

    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "best_model.h5")
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path, monitor='val_sparse_categorical_accuracy', save_best_only=True, mode='max', verbose=1
    )
    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args.out, "tensorboard"))
    early = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=6, restore_best_weights=True)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt_cb, tb_cb, early],
        class_weight=class_weights
    )

    # save final model & history
    final_path = os.path.join(args.out, "final_model.h5")
    model.save(final_path)
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history.history, fh, indent=2)
    print("Saved final model to:", final_path)

    # ---- Evaluate on test ----
    y_true = []
    y_prob = []
    for xb, yb in tqdm(test_ds, desc="Predicting test"):
        p = model.predict(xb)
        y_prob.extend(p.tolist())
        y_true.extend(yb.numpy().tolist())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = np.argmax(y_prob, axis=1)

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=labels_sorted, digits=4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # per-class AUC
    try:
        y_true_bin = label_binarize(y_true, classes=list(range(len(labels_sorted))))
        aucs = {}
        for i, lab in enumerate(labels_sorted):
            try:
                aucs[lab] = float(roc_auc_score(y_true_bin[:, i], y_prob[:, i]))
            except:
                aucs[lab] = float('nan')
        print("\nPer-class AUCs:", aucs)
    except Exception as e:
        print("AUC computation skipped:", e)

    print("\nRun saved to:", args.out)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="Path to index_with_splits.csv")
    p.add_argument("--out", default="runs/nano_v3", help="Output folder")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use_class_weights", action="store_true", help="Compute and use class weights from train set")
    args = p.parse_args()
    main(args)
