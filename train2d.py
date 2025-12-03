#!/usr/bin/env python3
"""
train2d_pytorch.py

Slice-wise 2D CNN in PyTorch with learnable per-channel window parameters.

Assumes index CSV (no splits) with columns: npy_path,label,study
(preprocessed volumes: (Z,H,W,C) with channels normalized to [0,1])

Script will:
 - read index.csv
 - stratify into train/val/test
 - build slice-wise datasets and dataloaders (multi-worker)
 - train tiny 2D CNN on slices
 - learn per-channel window center/width (applied before CNN)
 - evaluate at study-level by averaging slice probs per study
"""

import os
import argparse
import random
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ------------------------
# Reproducibility
# ------------------------
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn deterministic might slow down
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------
# Dataset (slice-wise)
# ------------------------
class SliceIndexDataset(Dataset):
    """
    Builds an index of (npy_path, slice_idx, label_idx, study) for all slices of all volumes.
    Returns single slices shaped (C,H,W) as torch.float32.
    """
    def __init__(self, df, label_map):
        """
        df: DataFrame with columns npy_path,label,study
        label_map: dict label->int
        """
        self.entries = []  # list of tuples (npy_path, slice_idx, label_idx, study)
        for _, row in df.iterrows():
            p = row['npy_path']
            lbl = label_map[row['label']]
            study = row['study']
            try:
                vol = np.load(p, mmap_mode='r')
            except Exception as e:
                print(f"[WARN] failed loading {p}: {e}")
                continue
            Z = vol.shape[0]
            for z in range(Z):
                self.entries.append((p, int(z), int(lbl), study))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        p, z, lbl, study = self.entries[idx]
        vol = np.load(p)  # load whole volume (fast if cached by OS)
        slice_img = vol[z]  # (H,W,C)
        # convert to C,H,W and to float32
        arr = np.asarray(slice_img, dtype=np.float32)
        # If channel-last and C maybe 1,2,... ensure channel-first
        if arr.ndim == 2:
            arr = arr[..., None]
        arr = np.transpose(arr, (2,0,1))  # C,H,W
        return torch.from_numpy(arr), torch.tensor(lbl, dtype=torch.long), p, z, study

# ------------------------
# Study dataset for evaluation (returns full volume)
# ------------------------
class StudyVolumeDataset(Dataset):
    def __init__(self, df, label_map):
        self.rows = df.to_dict('records')
        self.label_map = label_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        vol = np.load(r['npy_path']).astype(np.float32)  # (Z,H,W,C)
        if vol.ndim == 3:
            vol = vol[..., None]
        # convert to torch (Z,C,H,W)
        vol_t = torch.from_numpy(np.transpose(vol, (0,3,1,2))).float()
        label = self.label_map[r['label']]
        return vol_t, int(label), r['study'], r['npy_path']

# ------------------------
# Learnable windowing layer
# ------------------------
class LearnableWindow(nn.Module):
    """
    Per-channel learnable center and width in normalized [0,1] space.
    Input assumed in [0,1]. Output in [0,1] after windowing.
    """
    def __init__(self, n_channels, init_center=0.5, init_width=0.5, eps=1e-3):
        super().__init__()
        self.n = n_channels
        # center: parametrize with logit -> logits such that sigmoid(logits)=center
        init_center_tensor = torch.full((n_channels,), float(init_center), dtype=torch.float32)
        # nudge away from exact 0/1 for numerical stability, then logit
        init_center_tensor = (init_center_tensor * 0.999) + 0.0005
        self.center_logits = nn.Parameter(torch.logit(init_center_tensor))

        # width: parametrize via inverse softplus (we'll store a value that softplus maps to desired init),
        # but simpler: store log(init_width) as a learnable unconstrained param and map via softplus->(0,inf)->scale to (0,1]
        init_width_tensor = torch.full((n_channels,), float(init_width), dtype=torch.float32)
        # make a small-clamped positive value to avoid log(0)
        init_width_tensor = init_width_tensor.clamp(min=1e-6, max=0.999)
        # use log to initialize an unconstrained parameter
        self.width_un = nn.Parameter(torch.log(init_width_tensor))

        self.eps = eps

    def forward(self, x):
        """
        x: (B, C, H, W) in [0,1]
        returns same shape
        """
        centers = torch.sigmoid(self.center_logits)  # (C,)
        widths = F.softplus(self.width_un)            # positive
        widths = widths / (1.0 + widths)              # scale to (0,1)
        widths = widths.clamp(min=self.eps, max=1.0)

        c = centers.view(1, -1, 1, 1)
        w = widths.view(1, -1, 1, 1)

        lo = c - 0.5 * w
        hi = c + 0.5 * w

        x_clamped = torch.max(torch.min(x, hi), lo)
        out = (x_clamped - lo) / (w + 1e-12)
        out = out.clamp(0.0, 1.0)
        return out

# lw = LearnableWindow(n_channels=4, init_center=0.5, init_width=0.5)
# print("centers (sigmoid):", torch.sigmoid(lw.center_logits).detach().cpu().numpy())
# print("widths (mapped):", (F.softplus(lw.width_un)/(1+F.softplus(lw.width_un))).detach().cpu().numpy())

# ------------------------
# Tiny 2D CNN (per-slice)
# ------------------------
import torch
import torch.nn as nn

class TinySliceCNN(nn.Module):
    def __init__(self, in_channels=4, n_classes=3, embedding_dim=64):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.features = nn.Sequential(
            conv_block(in_channels, 32),
            nn.MaxPool2d(2),   # 64→32

            conv_block(32, 64),
            nn.MaxPool2d(2),   # 32→16

            conv_block(64, 128),
            nn.MaxPool2d(2),   # 16→8

            conv_block(128, 256),
            nn.AdaptiveAvgPool2d(1)  # (B,256,1,1)
        )

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        feat = self.features(x)      # (B,256,1,1)
        feat = self.embedding(feat)  # (B,embedding_dim)
        logits = self.classifier(feat)
        return logits, feat

# ------------------------
# Utilities
# ------------------------
def make_splits(df, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    # initial train/temp
    df_train, df_temp = train_test_split(df, test_size=(1.0 - train_ratio), stratify=df['label'], random_state=seed)
    # split temp into val/test
    val_frac = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(df_temp, test_size=(1.0 - val_frac), stratify=df_temp['label'], random_state=seed)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)

def collate_slices(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.stack([b[1] for b in batch], dim=0)
    # other meta not used in training
    return imgs, labels

# ------------------------
# Training and eval loops
# ------------------------
def train_epoch(model, win_layer, loader, optimizer, device, criterion):
    model.train()
    win_layer.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels, *_ in pbar:
        imgs = imgs.to(device)        # (B, C, H, W)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # apply learnable window
            imgs_w = win_layer(imgs)
            logits, _ = model(imgs_w)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += int((preds == labels).sum().item())
        total += imgs.size(0)
        pbar.set_postfix(loss=total_loss/total, acc=total_correct/total)
    return total_loss/total, total_correct/total

def validate_slice(model, win_layer, loader, device, criterion):
    model.eval()
    win_layer.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for imgs, labels, *_ in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs_w = win_layer(imgs)
            logits, _ = model(imgs_w)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * imgs.size(0)
            preds = logits.argmax(dim=1)
            total_correct += int((preds == labels).sum().item())
            total += imgs.size(0)
            pbar.set_postfix(loss=total_loss/total, acc=total_correct/total)
    return total_loss/total, total_correct/total

def eval_study_level(slice_model, win_layer, study_df, label_map, device, batch_size=32, num_workers=2):
    """
    For each study, predict all slices, average probabilities and compute study-level preds.
    Returns true_labels, pred_labels, pred_probs (list of arrays).
    """
    slice_model.eval()
    win_layer.eval()
    ds = StudyVolumeDataset(study_df, label_map)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=max(1, num_workers), pin_memory=True)
    y_true = []
    y_pred = []
    y_prob = []
    with torch.no_grad():
        for vol_t, label, study, path in tqdm(loader, desc="study-eval"):
            # vol_t: (1, Z, C, H, W)
            vol_t = vol_t.squeeze(0).to(device)  # (Z, C, H, W)
            Z = vol_t.shape[0]
            # predict in batches of slices
            probs_list = []
            for i in range(0, Z, batch_size):
                batch = vol_t[i:i+batch_size]   # (b, C, H, W)
                batch_w = win_layer(batch)
                logits, _ = slice_model(batch_w)
                probs = F.softmax(logits, dim=1).cpu().numpy()  # (b, n_classes)
                probs_list.append(probs)
            probs_all = np.vstack(probs_list)  # (Z, n_classes)
            mean_prob = probs_all.mean(axis=0)
            pred = int(np.argmax(mean_prob))
            y_true.append(int(label))
            y_pred.append(pred)
            y_prob.append(mean_prob)
    return np.array(y_true), np.array(y_pred), np.vstack(y_prob)

# ------------------------
# Main
# ------------------------
def main(args):
    seed_everything(args.seed)

    # Read index
    df = pd.read_csv(args.index)
    # minimal sanity
    df = df[df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Index CSV is empty or paths not found.")

    labels_sorted = sorted(df['label'].unique())
    label_map = {lab: i for i, lab in enumerate(labels_sorted)}
    print("Labels:", labels_sorted)

    # Split
    train_df, val_df, test_df = make_splits(df, seed=args.seed, train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)
    print("Split sizes (studies):", len(train_df), len(val_df), len(test_df))

    # Build datasets (slice-wise)
    train_ds = SliceIndexDataset(train_df, label_map)
    val_ds = SliceIndexDataset(val_df, label_map) if len(val_df)>0 else None

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_slices, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_slices, pin_memory=True) if val_ds is not None else None

    # Create model and window layer
    # infer channels from first sample:
    sample_vol = np.load(df.iloc[0]['npy_path'])
    C = sample_vol.shape[-1] if sample_vol.ndim==4 else 1
    print("Input channels:", C)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    slice_model = TinySliceCNN(in_channels=C, n_classes=len(labels_sorted), embedding_dim=args.embedding).to(device)
    win_layer = LearnableWindow(n_channels=C, init_center=0.5, init_width=0.5).to(device)

    # optimizer includes both model and window params
    optimizer = torch.optim.Adam(list(slice_model.parameters()) + list(win_layer.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # class weights optional: we compute on slice-level (each study contributes Z slices)
    if args.use_class_weights:
        y_train = []
        for _, r in train_df.iterrows():
            Z = np.load(r['npy_path']).shape[0]
            y_train.extend([label_map[r['label']]] * Z)
        from sklearn.utils.class_weight import compute_class_weight
        cw = compute_class_weight('balanced', classes=np.arange(len(labels_sorted)), y=y_train)
        # we will integrate cw into criterion by using weight param
        class_weights_tensor = torch.tensor(cw, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
        print("Using class weights:", cw)

    best_val_acc = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_acc = train_epoch(slice_model, win_layer, train_loader, optimizer, device, criterion)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

        if val_loader is not None:
            val_loss, val_acc = validate_slice(slice_model, win_layer, val_loader, device, criterion)
            history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
            print(f"Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}")
            # simple checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(args.out, exist_ok=True)
                torch.save({'model_state': slice_model.state_dict(),
                            'win_state': win_layer.state_dict()},
                           os.path.join(args.out, 'best_model.pth'))
        else:
            print(f"Train loss {train_loss:.4f} acc {train_acc:.4f}")

    # final save
    os.makedirs(args.out, exist_ok=True)
    torch.save({'model_state': slice_model.state_dict(),
                'win_state': win_layer.state_dict(),
                'label_map': label_map},
               os.path.join(args.out, 'final_model.pth'))

    # Save history
    with open(os.path.join(args.out, 'history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)

    # Study-level evaluation on test set
    if len(test_df) > 0:
        y_true, y_pred, y_prob = eval_study_level(slice_model, win_layer, test_df, label_map, device, batch_size=args.eval_batch, num_workers=min(8,args.workers))
        print("\nStudy-level Classification Report:")
        print(classification_report(y_true, y_pred, target_names=labels_sorted, digits=4))
        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred))
        try:
            y_bin = label_binarize(y_true, classes=list(range(len(labels_sorted))))
            aucs = {}
            for i, lab in enumerate(labels_sorted):
                try:
                    aucs[lab] = float(roc_auc_score(y_bin[:, i], y_prob[:, i]))
                except:
                    aucs[lab] = float('nan')
            print("\nPer-class AUCs:", aucs)
        except Exception as e:
            print("AUC skipped:", e)

    print("Done. Artifacts saved to:", args.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="index.csv from preprocessing (npy_path,label,study)")
    p.add_argument("--out", default="runs/pt2d", help="output folder")
    p.add_argument("--batch", type=int, default=32, help="slice batch size")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8, help="DataLoader num_workers")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.7)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--embedding", type=int, default=16)
    p.add_argument("--use_class_weights", action="store_true")
    p.add_argument("--eval_batch", type=int, default=64)
    args = p.parse_args()
    main(args)
