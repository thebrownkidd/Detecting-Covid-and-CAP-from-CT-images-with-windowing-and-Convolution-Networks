#!/usr/bin/env python3
"""
train2d_pytorch.py — UPDATED FOR 3-CHANNEL INPUT

Uses slice-wise 2D CNN + learnable windowing.
Assumes preprocessed volumes shaped (Z, H, W, 3).

Works with DataPrep_v3_3_multithread_tightwindows.py
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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------
# Dataset (slice-wise)
# ------------------------
class SliceIndexDataset(Dataset):
    """
    Returns slices as (C,H,W)
    """
    def __init__(self, df, label_map):
        self.entries = []
        for _, row in df.iterrows():
            p = row['npy_path']
            lbl = label_map[row['label']]
            study = row['study']
            try:
                vol = np.load(p, mmap_mode='r')
            except:
                continue
            Z = vol.shape[0]
            for z in range(Z):
                self.entries.append((p, z, lbl, study))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        p, z, lbl, study = self.entries[idx]
        vol = np.load(p)  # (Z,H,W,C)
        slice_img = vol[z]  # (H,W,3)
        arr = slice_img.astype(np.float32)

        # channel-first
        arr = np.transpose(arr, (2,0,1))
        return torch.from_numpy(arr), torch.tensor(lbl, dtype=torch.long), p, z, study


# ------------------------
# Study-level dataset
# ------------------------
class StudyVolumeDataset(Dataset):
    def __init__(self, df, label_map):
        self.rows = df.to_dict('records')
        self.label_map = label_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        vol = np.load(r['npy_path']).astype(np.float32)  # (Z,H,W,3)
        vol_t = torch.from_numpy(np.transpose(vol, (0,3,1,2)))  # (Z,3,H,W)
        label = self.label_map[r['label']]
        return vol_t, label, r['study'], r['npy_path']


# ------------------------
# Learnable Window Layer
# ------------------------
class LearnableWindow(nn.Module):
    """
    Learnable per-channel window center and width in normalized [0,1] space.
    """
    def __init__(self, n_channels, init_center=0.5, init_width=0.5, eps=1e-3):
        super().__init__()
        self.n = n_channels

        init_center_tensor = torch.full((n_channels,), float(init_center))
        init_center_tensor = init_center_tensor.clamp(0.001, 0.999)
        self.center_logits = nn.Parameter(torch.logit(init_center_tensor))

        init_width_tensor = torch.full((n_channels,), float(init_width)).clamp(1e-6, 0.999)
        self.width_un = nn.Parameter(torch.log(init_width_tensor))

        self.eps = eps

    def forward(self, x):
        centers = torch.sigmoid(self.center_logits)
        widths = F.softplus(self.width_un)
        widths = widths / (1+widths)
        widths = widths.clamp(self.eps, 1.0)

        c = centers.view(1, -1, 1, 1)
        w = widths.view(1, -1, 1, 1)

        lo = c - 0.5*w
        hi = c + 0.5*w

        x_clamped = torch.clamp(x, lo, hi)
        out = (x_clamped - lo) / (w + 1e-12)
        return out.clamp(0.0, 1.0)


# ------------------------
# Tiny 2D CNN
# ------------------------
class TinySliceCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=3, embedding_dim=16):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 12, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)

        # final embedding dimensionality
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = self.conv1(x)   # (B,8,H,W)
        x = self.pool(x)    # H/2

        x = self.conv2(x)   # (B,12,H/2,W/2)
        x = self.pool(x)    # H/4

        x = self.conv3(x)   # (B,24,H/4,W/4)
        x = self.pool(x)    # H/8

        x = self.gap(x)     # (B,24,1,1)
        x = self.embedding(x)  # (B, embedding_dim)
        logits = self.classifier(x)
        return logits, x


# ------------------------
# Utility: Data splits
# ------------------------
def make_splits(df, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    df_train, df_temp = train_test_split(df, test_size=(1-train_ratio),
                                         stratify=df["label"], random_state=seed)
    val_frac = val_ratio / (val_ratio + test_ratio)
    df_val, df_test = train_test_split(df_temp, test_size=(1-val_frac),
                                       stratify=df_temp["label"], random_state=seed)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)


def collate_slices(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.stack([b[1] for b in batch])
    return imgs, labels


# ------------------------
# Train + Validate loops
# ------------------------
def train_epoch(model, win_layer, loader, optimizer, device, criterion):
    model.train(); win_layer.train()
    total_loss = 0; total_correct = 0; total = 0

    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        imgs_w = win_layer(imgs)
        logits, _ = model(imgs_w)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return total_loss/total, total_correct/total


def validate(model, win_layer, loader, device, criterion):
    model.eval(); win_layer.eval()
    total_loss = 0; total_correct = 0; total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="val", leave=False):
            imgs = imgs.to(device)
            labels = labels.to(device)

            imgs_w = win_layer(imgs)
            logits, _ = model(imgs_w)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            total_correct += (logits.argmax(1) == labels).sum().item()
            total += imgs.size(0)

    return total_loss/total, total_correct/total


# ------------------------
# Study-level evaluation
# ------------------------
def eval_study_level(model, win_layer, df_test, label_map, device, batch_size=64, workers=4):
    ds = StudyVolumeDataset(df_test, label_map)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers)

    y_true = []; y_pred = []; y_prob = []

    model.eval(); win_layer.eval()

    with torch.no_grad():
        for vol_t, label, study, path in tqdm(loader, desc="study-eval"):
            vol_t = vol_t.squeeze(0).to(device)  # (Z,3,H,W)
            Z = vol_t.shape[0]
            probs_all = []

            for i in range(0, Z, batch_size):
                batch = vol_t[i:i+batch_size]
                batch_w = win_layer(batch)
                logits, _ = model(batch_w)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                probs_all.append(probs)

            probs_all = np.vstack(probs_all)
            mean_prob = probs_all.mean(axis=0)

            y_true.append(label)
            y_pred.append(np.argmax(mean_prob))
            y_prob.append(mean_prob)

    return np.array(y_true), np.array(y_pred), np.vstack(y_prob)


# ------------------------
# Main
# ------------------------
def main(args):
    seed_everything(args.seed)

    df = pd.read_csv(args.index)
    df = df[df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No valid entries in index.")

    labels_sorted = sorted(df['label'].unique())
    label_map = {lab:i for i,lab in enumerate(labels_sorted)}
    print("Labels:", labels_sorted)

    train_df, val_df, test_df = make_splits(df, seed=args.seed)
    print("Split sizes:", len(train_df), len(val_df), len(test_df))

    train_ds = SliceIndexDataset(train_df, label_map)
    val_ds = SliceIndexDataset(val_df, label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_slices)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.workers, collate_fn=collate_slices)

    C = 3  # IMPORTANT: 3-channel input
    print("Input channels =", C)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    model = TinySliceCNN(in_channels=C, n_classes=len(labels_sorted),
                         embedding_dim=args.embedding).to(device)
    win_layer = LearnableWindow(n_channels=C).to(device)

    optimizer = torch.optim.Adam(list(model.parameters()) +
                                 list(win_layer.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0

    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, win_layer, train_loader,
                                            optimizer, device, criterion)
        val_loss, val_acc = validate(model, win_layer, val_loader,
                                     device, criterion)
        print(f"Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model':model.state_dict(),
                        'win':win_layer.state_dict(),
                        'label_map':label_map},
                        os.path.join(args.out, "best.pth"))

    # Final save
    torch.save({'model':model.state_dict(),
                'win':win_layer.state_dict(),
                'label_map':label_map},
                os.path.join(args.out, "final.pth"))

    # Study-level eval
    y_true, y_pred, y_prob = eval_study_level(model, win_layer, test_df,
                                              label_map, device)
    print("\nStudy-level report:")
    print(classification_report(y_true, y_pred, target_names=labels_sorted))

    print("Confusion:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True)
    p.add_argument("--out", default="runs/pt2d")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--embedding", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    main(args)
