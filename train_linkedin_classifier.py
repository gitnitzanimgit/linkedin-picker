#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EfficientNet-B0 fine-tune for LinkedIn head-shot quality
with an explicit 3 × FP  +  1 × FN business-cost objective.

Core pipeline identical to the original ResNet-18 version; only the backbone
and its un-/freeze rules were swapped.

──────────────────────────────────────────────────────────────────────────────
Key points
──────────
* Pre-trained backbone:  EfficientNet-B0 (torchvision v0.22 weights)
* Phase-1:  train the new classifier head only (4 epochs, lr=1e-3)
* Phase-2:  unfreeze **features.6** block + classifier (16 epochs, lr=5e-5)
* Validation-time threshold search minimises (3·FP + 1·FN)
* Saves state-dict, image-stats and best threshold to `linkedin_efficientb0_cost_min.pth`
"""

from pathlib import Path
import argparse, random, math, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, confusion_matrix, fbeta_score
)

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

# ─── CONFIG ───────────────────────────────────────────────────
SEED            = 42
BATCH_SIZE      = 32
LR_HEAD         = 1e-3
LR_BACKBONE     = 5e-5
EPOCHS_PHASE1   = 4
EPOCHS_PHASE2   = 16
PATIENCE        = 5
IMG_MEAN        = [0.485, 0.456, 0.406]
IMG_STD         = [0.229, 0.224, 0.225]

NEG_WEIGHT      = 3.0
POS_WEIGHT      = 1.0
FP_COST         = 3
FN_COST         = 1
BETA_F          = 0.5        # F-0.5 emphasises precision

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ─── DATASET / DATALOADER ────────────────────────────────────
class LinkedInDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.img_dir / row.image_name).convert("RGB")
        label = torch.tensor(row.label, dtype=torch.float32)
        return self.transform(img), label

def make_loader(df, img_dir, transform, shuffle=False):
    ds = LinkedInDataset(df, img_dir, transform)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=4, pin_memory=False)      # pin_memory off on M-series

# ─── LOSS ────────────────────────────────────────────────────
def weighted_bce(logits, targets):
    w = torch.where(targets == 1, POS_WEIGHT, NEG_WEIGHT).to(logits.device)
    return nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=w, reduction='mean'
    )

# ─── TRAIN / EVAL ────────────────────────────────────────────
def train_one_epoch(model, loader, optim, device):
    model.train()
    loss_sum, preds, labels = 0., [], []
    for x, y in loader:
        x, y = x.to(device), y.unsqueeze(1).to(device)
        optim.zero_grad()
        logit = model(x)
        loss = weighted_bce(logit, y)
        loss.backward(); optim.step()

        loss_sum += loss.item() * x.size(0)
        preds.append(torch.sigmoid(logit.detach()).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).squeeze(), torch.cat(labels).squeeze()
    return loss_sum/len(loader.dataset), roc_auc_score(labels, preds)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_sum, preds, labels = 0., [], []
    for x, y in loader:
        x, y = x.to(device), y.unsqueeze(1).to(device)
        logit = model(x)
        loss_sum += weighted_bce(logit, y).item() * x.size(0)
        preds.append(torch.sigmoid(logit).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).squeeze(), torch.cat(labels).squeeze()
    return loss_sum/len(loader.dataset), roc_auc_score(labels, preds), preds, labels

# ─── MAIN ────────────────────────────────────────────────────
def main(root="training_data/all_data"):
    root = Path(root)
    out_path = Path("linkedin_efficientb0_cost_min.pth")

    # 1 – load & downsample 444 / 444 for fast iteration
    df_full = pd.read_csv(root / "labels.csv")
    good = df_full[df_full["suitable"] == 1].sample(444, random_state=SEED)
    bad  = df_full[df_full["not-suitable"] == 1].sample(444, random_state=SEED)
    df   = pd.concat([good, bad]).reset_index(drop=True)
    df["label"] = (df["suitable"] == 1).astype(int)

    # 2 – stratified 70 / 15 / 15 split
    train_df, tmp_df = train_test_split(df, test_size=0.30,
                                        stratify=df.label, random_state=SEED)
    val_df, test_df  = train_test_split(tmp_df, test_size=0.50,
                                        stratify=tmp_df.label, random_state=SEED)

    # 3 – transforms
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD),
        transforms.RandomErasing(p=0.25)
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize(IMG_MEAN, IMG_STD)
    ])

    img_dir = root / "images"
    train_dl = make_loader(train_df, img_dir, train_tf, shuffle=True)
    val_dl   = make_loader(val_df,   img_dir, eval_tf)
    test_dl  = make_loader(test_df,  img_dir, eval_tf)

    # 4 – model: EfficientNet-B0
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    # replace classifier head (classifier = Sequential[Dropout, Linear])
    in_f = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_f, 1)
    model = model.to(device)

    # ── Phase-1: train head only ─────────────────────────────
    for p in model.parameters(): p.requires_grad = False
    model.classifier[1].weight.requires_grad = True
    model.classifier[1].bias.requires_grad  = True
    opt_head = torch.optim.AdamW(model.classifier[1].parameters(), lr=LR_HEAD)

    best_state, best_vloss, no_imp = None, math.inf, 0
    print("\nPhase 1 (head only)")
    for e in range(EPOCHS_PHASE1):
        tr_loss,_ = train_one_epoch(model, train_dl, opt_head, device)
        vl_loss, vl_auc, *_ = evaluate(model, val_dl, device)
        print(f"[{e+1}/{EPOCHS_PHASE1}] tr_loss={tr_loss:.4f} "
              f"val_loss={vl_loss:.4f} val_auc={vl_auc:.3f}")
        if vl_loss < best_vloss - 1e-4:
            best_vloss, no_imp, best_state = vl_loss, 0, model.state_dict()
        else:
            no_imp += 1

    # ── Phase-2: unfreeze last block (features.6) + classifier ──────────────
    for n,p in model.named_parameters():
        if n.startswith("features.6") or n.startswith("classifier"):
            p.requires_grad = True
    opt_back = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LR_BACKBONE)

    print("\nPhase 2 (fine-tune features.6 + head)")
    for e in range(EPOCHS_PHASE2):
        tr_loss,_ = train_one_epoch(model, train_dl, opt_back, device)
        vl_loss, vl_auc, *_ = evaluate(model, val_dl, device)
        print(f"[{e+1}/{EPOCHS_PHASE2}] tr_loss={tr_loss:.4f} "
              f"val_loss={vl_loss:.4f} val_auc={vl_auc:.3f}")
        if vl_loss < best_vloss - 1e-4:
            best_vloss, no_imp, best_state = vl_loss, 0, model.state_dict()
        else:
            no_imp += 1
            if no_imp >= PATIENCE:
                print("Early stopping."); break

    model.load_state_dict(best_state)

    # 5 – threshold search (3·FP + 1·FN)
    _,_, val_preds, val_labels = evaluate(model, val_dl, device)
    thr_grid = np.linspace(0,1,1001)
    best_thr, best_cost = 0., float('inf')
    for t in thr_grid:
        pred = (val_preds >= t)
        fp = int(((pred==1)&(val_labels==0)).sum())
        fn = int(((pred==0)&(val_labels==1)).sum())
        cost = FP_COST*fp + FN_COST*fn
        if cost < best_cost: best_cost, best_thr = cost, t
    print(f"Optimal threshold = {best_thr:.3f}  (val cost={best_cost})")

    # 6 – validation metrics @ threshold
    val_bin = (val_preds >= best_thr).int()
    val_cost = FP_COST*int(((val_bin==1)&(val_labels==0)).sum()) + \
               FN_COST*int(((val_bin==0)&(val_labels==1)).sum())
    print(f"Val acc={accuracy_score(val_labels,val_bin):.3f} "
          f"F-0.5={fbeta_score(val_labels,val_bin,beta=BETA_F):.3f} "
          f"cost={val_cost}")

    # 7 – test metrics
    tst_loss,tst_auc,tst_preds,tst_labels = evaluate(model,test_dl,device)
    tst_bin  = (tst_preds>=best_thr).int()
    tst_cost = FP_COST*int(((tst_bin==1)&(tst_labels==0)).sum()) + \
               FN_COST*int(((tst_bin==0)&(tst_labels==1)).sum())
    print("\nTest results")
    print(f"loss={tst_loss:.4f} AUC={tst_auc:.3f} "
          f"acc={accuracy_score(tst_labels,tst_bin):.3f} "
          f"F-0.5={fbeta_score(tst_labels,tst_bin,beta=BETA_F):.3f} "
          f"cost={tst_cost}")
    print("confusion:", confusion_matrix(tst_labels,tst_bin).tolist())

    # 8 – save checkpoint
    torch.save({
        "state_dict": model.state_dict(),
        "img_size": 224,
        "mean": IMG_MEAN,
        "std":  IMG_STD,
        "threshold": float(best_thr),
        "fp_cost": FP_COST,
        "fn_cost": FN_COST
    }, out_path)
    print(f"\n✓ Saved → {out_path.resolve()}")

# ─── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="training_data/all_data",
                    help="dir with images/ and labels.csv")
    main(ap.parse_args().root)
