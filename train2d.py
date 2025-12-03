#!/usr/bin/env python3
"""
train2d_pytorch_weighted.py

Slice-wise 2D CNN in PyTorch with:
 - learnable per-channel windowing (tight init)
 - ultra-tiny CNN (~6k params) returning (logits, embedding)
 - learnable slice-level weighting for study aggregation (trained on val)
 - study-level evaluation uses weighted average of slice probs

Assumes index CSV (no splits) with columns: npy_path,label,study
(preprocessed volumes: (Z,H,W,3) with values in [0,1])
"""
import os
import argparse
import random
import json
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
# Datasets
# ------------------------
class SliceIndexDataset(Dataset):
    """Slice-level dataset: returns (C,H,W), label, path, z, study"""
    def __init__(self, df, label_map):
        self.entries = []
        for _, row in df.iterrows():
            p = row['npy_path']
            lbl = label_map[row['label']]
            study = row['study']
            try:
                vol = np.load(p, mmap_mode='r')
            except Exception as e:
                print("[WARN] load failed:", p, e)
                continue
            Z = vol.shape[0]
            for z in range(Z):
                self.entries.append((p, int(z), int(lbl), study))

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        p, z, lbl, study = self.entries[idx]
        vol = np.load(p)                      # (Z,H,W,C)
        slice_img = vol[z]                    # (H,W,3)
        arr = slice_img.astype(np.float32)
        arr = np.transpose(arr, (2,0,1))      # (C,H,W)
        return torch.from_numpy(arr), torch.tensor(lbl, dtype=torch.long), p, z, study

class StudyVolumeDataset(Dataset):
    """Study-level dataset for eval / weight-layer updates"""
    def __init__(self, df, label_map):
        self.rows = df.to_dict('records')
        self.label_map = label_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        vol = np.load(r['npy_path']).astype(np.float32)   # (Z,H,W,3)
        vol_t = torch.from_numpy(np.transpose(vol, (0,3,1,2)))  # (Z,3,H,W)
        label = self.label_map[r['label']]
        return vol_t, int(label), r['study'], r['npy_path']

# ------------------------
# LearnableWindow (tight init)
# ------------------------
class LearnableWindow(nn.Module):
    """
    Per-channel learnable center & width in normalized [0,1] space.
    Tight initialization to avoid early overexposure.
    """
    def __init__(self, n_channels, init_centers=None, init_widths=None, eps=1e-4):
        super().__init__()
        self.n = n_channels
        self.eps = eps

        if init_centers is None:
            # radiologist-tight defaults for (raw, lung, soft)
            default = [0.45, 0.50, 0.55]
            init_centers = default[:n_channels]
        if init_widths is None:
            default_w = [0.15, 0.20, 0.25]
            init_widths = default_w[:n_channels]

        centers = torch.tensor(init_centers, dtype=torch.float32).clamp(1e-3, 0.999)
        widths = torch.tensor(init_widths, dtype=torch.float32).clamp(1e-6, 0.999)

        # parametrize center via logits (so sigmoid(center_logits) = center)
        self.center_logits = nn.Parameter(torch.logit(centers * 0.999 + 1e-6))
        # parametrize width via log (map -> positive through softplus later)
        self.width_un = nn.Parameter(torch.log(widths))

    def forward(self, x):
        # x: (B, C, H, W)
        centers = torch.sigmoid(self.center_logits)   # (C,)
        widths = F.softplus(self.width_un)            # positive
        widths = widths / (1.0 + widths)              # map to (0,1)
        widths = widths.clamp(self.eps, 1.0)

        c = centers.view(1, -1, 1, 1)
        w = widths.view(1, -1, 1, 1)
        lo = c - 0.5 * w
        hi = c + 0.5 * w

        x_clamped = torch.clamp(x, lo, hi)
        out = (x_clamped - lo) / (w + 1e-12)
        return out.clamp(0.0, 1.0)

# ------------------------
# Ultra-tiny CNN (6-7k params)
# ------------------------
class NanoCNN(nn.Module):
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
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv1(x)   # (B,8,H,W)
        x = self.pool(x)    # (B,8,H/2,W/2)

        x = self.conv2(x)   # (B,12,H/2,W/2)
        x = self.pool(x)    # (B,12,H/4,W/4)

        x = self.conv3(x)   # (B,24,H/4,W/4)
        x = self.pool(x)    # (B,24,H/8,W/8)

        x = self.gap(x)     # (B,24,1,1)
        emb = self.embedding(x)   # (B, embedding_dim)
        logits = self.classifier(emb)
        return logits, emb

# ------------------------
# Slice weighting module
# ------------------------
class SliceWeightingLayer(nn.Module):
    """Simple linear scorer over embeddings -> softmax over slices."""
    def __init__(self, embedding_dim):
        super().__init__()
        self.lin = nn.Linear(embedding_dim, 1, bias=True)

    def forward(self, embeddings):
        # embeddings: (Z, D) or (B, Z, D) ; we'll accept (Z,D)
        if embeddings.dim() == 3:
            # batch of studies not used — expecting (B=1, Z, D) not used here
            embeddings = embeddings.squeeze(0)
        scores = self.lin(embeddings).squeeze(-1)  # (Z,)
        weights = F.softmax(scores, dim=0)         # normalized across slices
        return weights

# ------------------------
# Utilities
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
# Train / validate (slice-level)
# ------------------------
def train_epoch(model, win_layer, loader, optimizer, device, criterion):
    model.train(); win_layer.train()
    total_loss = 0.0; total_correct = 0; total = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, labels, *_ in pbar:
        imgs = imgs.to(device); labels = labels.to(device)
        optimizer.zero_grad()
        imgs_w = win_layer(imgs)
        logits, _ = model(imgs_w)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * imgs.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += imgs.size(0)
        pbar.set_postfix(loss=total_loss/total, acc=total_correct/total)
    return total_loss/total, total_correct/total

def validate_slice(model, win_layer, loader, device, criterion):
    model.eval(); win_layer.eval()
    total_loss = 0.0; total_correct = 0; total = 0
    with torch.no_grad():
        pbar = tqdm(loader, desc="val", leave=False)
        for imgs, labels, *_ in pbar:
            imgs = imgs.to(device); labels = labels.to(device)
            imgs_w = win_layer(imgs)
            logits, _ = model(imgs_w)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * imgs.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total += imgs.size(0)
            pbar.set_postfix(loss=total_loss/total, acc=total_correct/total)
    return total_loss/total, total_correct/total

# ------------------------
# Study-level evaluation (uses learned slice weights)
# ------------------------
def eval_study_level(model, win_layer, weight_layer, df_test, label_map, device, batch_size=64, workers=4):
    ds = StudyVolumeDataset(df_test, label_map)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=workers)
    y_true = []; y_pred = []; y_prob = []

    model.eval(); win_layer.eval(); weight_layer.eval()
    with torch.no_grad():
        for vol_t, label, study, path in tqdm(loader, desc="study-eval"):
            vol_t = vol_t.squeeze(0).to(device)   # (Z,C,H,W)
            Z = vol_t.shape[0]
            emb_list = []; prob_list = []
            for i in range(0, Z, batch_size):
                batch = vol_t[i:i+batch_size]      # (b,C,H,W)
                batch_w = win_layer(batch)
                logits, emb = model(batch_w)
                probs = F.softmax(logits, dim=1)   # (b, n_classes)
                emb_list.append(emb.cpu())
                prob_list.append(probs.cpu())
            embeddings = torch.cat(emb_list, dim=0)   # (Z, D)
            probs_all = torch.cat(prob_list, dim=0)   # (Z, C)

            weights = weight_layer(embeddings.to(device))   # (Z,)
            final_prob = (weights.unsqueeze(1) * probs_all.to(device)).sum(dim=0)  # (C,)

            y_true.append(label)
            y_pred.append(int(final_prob.argmax().cpu().item()))
            y_prob.append(final_prob.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.vstack(y_prob)

# ------------------------
# Small function to update weight_layer using val set
# (we freeze model & window, compute embeddings with no_grad,
#  then backprop only through weight_layer using study-level CE loss)
# ------------------------
def update_weight_layer_on_val(model, win_layer, weight_layer, val_df, label_map, device, lr=1e-3):
    if len(val_df) == 0:
        return None
    model.eval(); win_layer.eval(); weight_layer.train()
    ds = StudyVolumeDataset(val_df, label_map)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    optimizer = torch.optim.Adam(weight_layer.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0; total = 0
    for vol_t, label, study, path in loader:
        vol = vol_t.squeeze(0).to(device)   # (Z,C,H,W)
        Z = vol.shape[0]
        emb_list = []; probs_list = []
        with torch.no_grad():
            for i in range(0, Z, 64):
                batch = vol[i:i+64]
                batch_w = win_layer(batch)
                logits, emb = model(batch_w)
                probs = F.softmax(logits, dim=1)   # (b,C)
                emb_list.append(emb.cpu())
                probs_list.append(probs.cpu())
        if len(emb_list) == 0:
            continue
        embeddings = torch.cat(emb_list, dim=0).to(device)   # (Z,D)
        probs_all = torch.cat(probs_list, dim=0).to(device)  # (Z,C)

        # forward through weight layer
        weights = weight_layer(embeddings)    # (Z,)
        final_prob = (weights.unsqueeze(1) * probs_all).sum(dim=0, keepdim=True)  # (1,C)
        target = torch.tensor([label], dtype=torch.long, device=device)
        loss = criterion(final_prob, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        total += 1

    if total == 0:
        return None
    return total_loss / total

# ------------------------
# Main
# ------------------------
def main(args):
    seed_everything(args.seed)

    df = pd.read_csv(args.index)
    df = df[df['npy_path'].apply(os.path.exists)].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("Index CSV has no valid entries.")

    labels_sorted = sorted(df['label'].unique())
    label_map = {lab:i for i,lab in enumerate(labels_sorted)}
    print("Labels:", labels_sorted)

    train_df, val_df, test_df = make_splits(df, seed=args.seed,
                                           train_ratio=args.train_ratio,
                                           val_ratio=args.val_ratio,
                                           test_ratio=args.test_ratio)
    print("Split sizes (studies):", len(train_df), len(val_df), len(test_df))

    train_ds = SliceIndexDataset(train_df, label_map)
    val_ds = SliceIndexDataset(val_df, label_map)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.workers, collate_fn=collate_slices, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=max(1,args.workers//2), collate_fn=collate_slices, pin_memory=True)

    C = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = NanoCNN(in_channels=C, n_classes=len(labels_sorted), embedding_dim=args.embedding).to(device)
    win_layer = LearnableWindow(n_channels=C).to(device)
    weight_layer = SliceWeightingLayer(embedding_dim=args.embedding).to(device)

    # optimizer trains model + window (slice-level). weight_layer updated separately on val (but include in optimizer if you want joint)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(win_layer.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[], 'weight_val_loss':[]}
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, win_layer, train_loader, optimizer, device, criterion)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

        val_loss, val_acc = validate_slice(model, win_layer, val_loader, device, criterion)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

        # update weighting layer using val set (small LR)
        weight_val_loss = update_weight_layer_on_val(model, win_layer, weight_layer, val_df, label_map, device, lr=args.weight_lr)
        history['weight_val_loss'].append(weight_val_loss if weight_val_loss is not None else 0.0)

        print(f"Train {train_loss:.4f}/{train_acc:.4f} | Val {val_loss:.4f}/{val_acc:.4f} | WvalLoss {weight_val_loss}")

        # Save checkpoint if val improves (use study-level eval with weights)
        y_true, y_pred, y_prob = eval_study_level(model, win_layer, weight_layer, val_df, label_map, device,
                                                 batch_size=args.eval_batch, workers=min(4,args.workers))
        from sklearn.metrics import accuracy_score
        study_val_acc = float(accuracy_score(y_true, y_pred)) if len(y_true)>0 else 0.0

        if study_val_acc > best_val:
            best_val = study_val_acc
            torch.save({
                'model_state': model.state_dict(),
                'win_state': win_layer.state_dict(),
                'weight_state': weight_layer.state_dict(),
                'label_map': label_map
            }, os.path.join(args.out, 'best.pth'))
            print("[✓] Saved best.pth (study-level val acc improved)")

    # final save
    torch.save({
        'model_state': model.state_dict(),
        'win_state': win_layer.state_dict(),
        'weight_state': weight_layer.state_dict(),
        'label_map': label_map
    }, os.path.join(args.out, 'final.pth'))
    with open(os.path.join(args.out, 'history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)

    # final study-level evaluation on test
    if len(test_df) > 0:
        y_true, y_pred, y_prob = eval_study_level(model, win_layer, weight_layer, test_df, label_map, device,
                                                 batch_size=args.eval_batch, workers=min(8,args.workers))
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

# ------------------------
# CLI
# ------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--index", required=True, help="index.csv with npy_path,label,study")
    p.add_argument("--out", default="runs/pt2d_weighted")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_lr", type=float, default=1e-3, help="LR for updating slice-weight layer on val")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--embedding", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--test_ratio", type=float, default=0.15)
    p.add_argument("--eval_batch", type=int, default=64)
    args = p.parse_args()
    main(args)
