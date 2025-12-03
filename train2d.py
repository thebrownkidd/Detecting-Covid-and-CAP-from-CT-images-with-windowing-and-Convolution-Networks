#!/usr/bin/env python3
"""
train2d_slice_classifier.py

Train a pure slice-wise 2D CNN (no augmentation). Aggregate slice predictions to study-level
by averaging slice probabilities at test time.

Index CSV must contain columns: npy_path,label,study,split
(preprocessed volumes shape: (Z,H,W,C) and Z must be consistent across dataset)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import random
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.experimental.AUTOTUNE

# -------------------------
# Slice generator (NO AUGMENTATION)
# -------------------------
def slice_generator(df, label_map):
    """Yield (slice_image, label_idx) for every slice in every study in df."""
    for _, row in df.iterrows():
        path = row['npy_path']
        label_idx = np.int32(label_map[row['label']])
        try:
            vol = np.load(path).astype(np.float32)   # (Z,H,W,C)
        except Exception as e:
            print(f"[WARN] failed to load {path}: {e}")
            continue
        Z = vol.shape[0]
        for i in range(Z):
            yield vol[i], label_idx


def build_slice_dataset(df, label_map, batch_size, repeat=False, shuffle=False):
    """Return tf.data.Dataset for slices and the sample shape (H,W,C) and slices_per_volume Z."""
    # sample
    sample_path = df.iloc[0]['npy_path']
    sample_vol = np.load(sample_path)
    Z = sample_vol.shape[0]
    H, W, C = sample_vol.shape[1], sample_vol.shape[2], (sample_vol.shape[3] if sample_vol.ndim == 4 else 1)

    out_types = (tf.float32, tf.int32)
    out_shapes = ((H, W, C), ())

    gen = lambda: slice_generator(df, label_map)
    ds = tf.data.Dataset.from_generator(gen, output_types=out_types, output_shapes=out_shapes)

    if shuffle:
        ds = ds.shuffle(buffer_size=4096, seed=SEED)

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, (Z, H, W, C)


# -------------------------
# Small 2D CNN (per-slice)
# -------------------------
def build_slice_cnn(input_shape=(64,64,4), embedding_dim=16):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(12, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    # small head for classification (we'll use a final classifier on top separately)
    x = tf.keras.layers.Dense(embedding_dim, activation="relu")(x)
    model = tf.keras.Model(inp, x, name="slice_cnn")
    return model


def build_classifier_model(H, W, C, embedding_dim, n_classes):
    """Model that takes a full volume and outputs class probabilities by averaging slice embeddings."""
    # We'll build a wrapper that expects (Z,H,W,C) but we'll train slice-wise model separately.
    # For training slices we only use slice_cnn standalone.
    inp = tf.keras.Input(shape=(None, H, W, C))  # None for Z (not used during slice training)
    slice_cnn = build_slice_cnn((H, W, C), embedding_dim)
    # TimeDistributed to get per-slice embeddings
    E = tf.keras.layers.TimeDistributed(slice_cnn)(inp)  # (B, Z, D)
    # Average embeddings across Z
    E_mean = tf.keras.layers.GlobalAveragePooling1D()(E)  # (B, D)
    x = tf.keras.layers.Dense(32, activation='relu')(E_mean)
    x = tf.keras.layers.Dropout(0.25)(x)
    out = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
    model = tf.keras.Model(inp, out)
    return model


# -------------------------
# Training / evaluation
# -------------------------
def main(args):
    df = pd.read_csv(args.index)
    # keep only existing paths
    df = df[df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Index CSV contains no existing .npy files.")

    labels_sorted = sorted(df['label'].unique())
    label_map = {lab: i for i, lab in enumerate(labels_sorted)}
    print("Labels:", labels_sorted)

    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df   = df[df['split'] == 'val'].reset_index(drop=True)
    test_df  = df[df['split'] == 'test'].reset_index(drop=True)

    if len(train_df) == 0:
        raise RuntimeError("No training studies found in index.")

    # build datasets (slice-level)
    train_ds, (Z_train, H, W, C) = build_slice_dataset(train_df, label_map, args.batch_size, repeat=True, shuffle=False)
    val_ds,   (Z_val,   _, _, _) = build_slice_dataset(val_df,   label_map, args.batch_size, repeat=False, shuffle=False) if len(val_df)>0 else (None, (Z_train,H,W,C))
    # note: val_ds not repeated
    print(f"Slices per volume (assumed uniform): {Z_train}, slice shape: ({H},{W},{C})")
    print(f"Train studies: {len(train_df)}, Val studies: {len(val_df)}, Test studies: {len(test_df)}")

    # build slice-level model (embedding + classifier head)
    # We'll create a slice model that outputs logits directly (so training is easier)
    slice_encoder = build_slice_cnn((H, W, C), embedding_dim=16)
    slice_input = tf.keras.Input(shape=(H, W, C))
    emb = slice_encoder(slice_input)           # (D,)
    logits = tf.keras.layers.Dense(len(labels_sorted), activation='softmax')(emb)
    slice_model = tf.keras.Model(slice_input, logits, name="slice_model")

    slice_model.summary()

    # compile
    slice_model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='sparse_acc')]
    )

    # class weights (train-level)
    class_weights = None
    if args.use_class_weights:
        y_train = []
        for _, row in train_df.iterrows():
            # each study contributes Z_train slices with same label
            y_train.extend([label_map[row['label']]] * Z_train)
        cw = compute_class_weight(class_weight='balanced', classes=np.arange(len(labels_sorted)), y=y_train)
        class_weights = {i: float(cw[i]) for i in range(len(labels_sorted))}
        print("Class weights:", class_weights)

    # callbacks
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, "best_slice_model.keras")
    ckpt = tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor='val_sparse_acc', save_best_only=True, mode='max', verbose=1)
    early = tf.keras.callbacks.EarlyStopping(monitor='val_sparse_acc', mode='max', patience=20, restore_best_weights=True, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_acc', factor=0.5, patience=6, verbose=1)

    # steps (slice-level)
    steps_per_epoch = max(1, (len(train_df) * Z_train) // args.batch_size)
    val_steps = max(1, (len(val_df) * Z_val) // args.batch_size) if len(val_df)>0 else None

    print("Steps per epoch (slices):", steps_per_epoch, "Validation steps:", val_steps)

    # fit on slices
    history = slice_model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds if val_ds is not None else None,
        validation_steps=val_steps if val_steps is not None else None,
        callbacks=[ckpt, reduce_lr, early],
        class_weight=class_weights
    )

    # save final slice model (native keras)
    final_path = os.path.join(args.out, "final_slice_model.keras")
    slice_model.save(final_path)
    with open(os.path.join(args.out, "history.json"), "w") as fh:
        json.dump(history.history, fh, indent=2)
    print("Saved slice model to:", final_path)

    # -------------------------
    # Study-level evaluation
    # -------------------------
    if len(test_df) == 0:
        print("No test studies provided; exiting.")
        return

    y_true = []
    y_pred = []
    y_prob = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Study eval"):
        path = row['npy_path']
        label_idx = label_map[row['label']]
        vol = np.load(path).astype(np.float32)  # (Z,H,W,C)
        # predict per-slice (batch slices for speed)
        # reshape to (-1,H,W,C)
        preds = slice_model.predict(vol, batch_size=args.batch_size, verbose=0)  # (Z, n_classes)
        # aggregate probabilities (mean)
        mean_prob = preds.mean(axis=0)
        pred_label = int(np.argmax(mean_prob))
        y_true.append(label_idx)
        y_pred.append(pred_label)
        y_prob.append(mean_prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.vstack(y_prob)

    print("\nStudy-level Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels_sorted, digits=4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    # per-class AUC (if possible)
    try:
        y_bin = label_binarize(y_true, classes=list(range(len(labels_sorted))))
        aucs = {}
        for i, lab in enumerate(labels_sorted):
            aucs[lab] = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
        print("\nPer-class AUCs:", aucs)
    except Exception as e:
        print("AUC skipped:", e)

    print("\nDone. Artifacts saved to:", args.out)


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="index csv with npy_path,label,study,split")
    p.add_argument("--out", default="runs/2d_slice", help="output directory")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--use_class_weights", action="store_true")
    args = p.parse_args()
    main(args)
