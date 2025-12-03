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
import os
import warnings

# Hide TF oneDNN and CPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all, 1 = INFO, 2 = WARNING, 3 = ERROR

# Suppress Python warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

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

def generator_from_df(df, label_map):
    """
    Yields (vol, label_idx) with NO augmentation.
    """
    for _, row in df.iterrows():
        try:
            vol = np.load(row['npy_path']).astype(np.float32)  # (Z,H,W,C)
        except Exception as e:
            print(f"[WARN] failed loading {row['npy_path']}: {e}")
            continue

        if vol.ndim == 3:  # safety
            vol = vol[..., None]

        yield vol, np.int32(label_map[row['label']])


def build_tf_dataset(df, label_map, batch):
    gen = lambda: generator_from_df(df, label_map)

    # Get sample shape
    for _, r in df.iterrows():
        sample = np.load(r['npy_path'])
        break

    if sample.ndim == 3:
        sample_shape = (sample.shape[0], sample.shape[1], sample.shape[2], 1)
    else:
        sample_shape = sample.shape

    output_types = (tf.float32, tf.int32)
    output_shapes = (sample_shape, ())

    ds = tf.data.Dataset.from_generator(
        gen,
        output_types=output_types,
        output_shapes=output_shapes
    )

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

    # Block 1: 8 filters only
    x = tf.keras.layers.Conv3D(8, 3, padding="same", activation='relu')(inp)
    x = tf.keras.layers.MaxPool3D(2)(x)

    # Block 2: 12 filters
    x = tf.keras.layers.Conv3D(12, 3, padding="same", activation='relu')(x)
    x = tf.keras.layers.MaxPool3D(2)(x)

    # Block 3: 16 filters
    x = tf.keras.layers.Conv3D(16, 3, padding="same", activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling3D()(x)

    # Very small dense head
    x = tf.keras.layers.Dense(32, activation="relu")(x)
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
    train_ds = build_tf_dataset(train_df, label_map, batch=args.batch_size)
    val_ds   = build_tf_dataset(val_df,   label_map, batch=args.batch_size)
    test_ds  = build_tf_dataset(test_df,  label_map, batch=args.batch_size)

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
    early = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', mode='max', patience=30, restore_best_weights=True)

    steps_per_epoch = len(train_df) // args.batch_size
    val_steps = len(val_df) // args.batch_size

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
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
    # print the length of the test dataset:
    
    print(test_ds.__len__())
    for xb, yb in enumerate(test_ds):
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
