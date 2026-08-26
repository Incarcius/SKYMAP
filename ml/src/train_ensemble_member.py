"""
Ensemble member training - identical setup to train.py (same architecture,
same data split, same augmentation) but with a different random seed, so
its errors are decorrelated from the main model. Averaging multiple
independently-initialized models' predictions is one of the most reliable
"free" accuracy techniques in segmentation - it doesn't fix systematic
blind spots (e.g. still won't have pretrained ImageNet features), but it
smooths out each individual model's noise/overfitting quirks.

Usage: python3 train_ensemble_member.py --member 1 --epochs 5 [--resume]
"""
import sys, time, argparse, os
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from model import AttentionResUNet
from dataset import PatchDataset, split_regions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

parser = argparse.ArgumentParser()
parser.add_argument('--member', type=int, required=True)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()

CKPT_PATH = f'{OUT_DIR}/ensemble_member_{args.member}.pt'
SEED = 200 + args.member
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(4)

rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
gt = np.array(Image.open(f'{DATA_DIR}/GT.png').convert('L')).astype(np.float32)
gt = (gt > 127).astype(np.float32)
H, W = gt.shape

# SAME split as the main model, so ensemble members are directly comparable
# and averageable on the same held-out region.
train_region, val_region = split_regions(H, W, val_fraction=0.2, axis='x', side='right')

train_ds = PatchDataset(rgb, gt, train_region, patch_size=128, stride=48, augment=True)
val_ds = PatchDataset(rgb, gt, val_region, patch_size=128, stride=32, augment=False)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

train_building_frac = gt[train_region[0]:train_region[1], train_region[2]:train_region[3]].mean()
pos_weight_val = min((1 - train_building_frac) / max(train_building_frac, 1e-4), 8.0)

model = AttentionResUNet(in_ch=3, out_ch=1, base=16)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))

start_epoch = 0
if args.resume and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])
    start_epoch = ckpt['epoch']
    print(f'Member {args.member}: resumed from epoch {start_epoch}', flush=True)
else:
    print(f'Member {args.member}: training from scratch (seed={SEED})', flush=True)


def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return 1 - ((2 * inter + eps) / (union + eps)).mean()


def combined_loss(pred, target):
    return 0.5 * bce(pred, target) + 0.5 * dice_loss(pred, target)


def iou_score(pred, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > thresh).float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = ((pred + target) > 0).float().sum(dim=(1, 2, 3))
    return ((inter + eps) / (union + eps)).mean().item()


t0 = time.time()
end_epoch = start_epoch + args.epochs
for epoch in range(start_epoch + 1, end_epoch + 1):
    model.train()
    tr_losses = []
    for img, msk in train_loader:
        opt.zero_grad()
        pred = model(img)
        loss = combined_loss(pred, msk)
        loss.backward()
        opt.step()
        tr_losses.append(loss.item())

    model.eval()
    val_losses, val_ious = [], []
    with torch.no_grad():
        for img, msk in val_loader:
            pred = model(img)
            val_losses.append(combined_loss(pred, msk).item())
            val_ious.append(iou_score(pred, msk))

    tr_l, va_l, va_iou = np.mean(tr_losses), np.mean(val_losses), np.mean(val_ious)
    scheduler.step(va_l)
    print(f'Member {args.member} | Epoch {epoch}/{end_epoch} | train_loss={tr_l:.4f} | '
          f'val_loss={va_l:.4f} | val_IoU={va_iou:.4f} | elapsed={time.time()-t0:.0f}s', flush=True)

    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, CKPT_PATH)

print(f'Member {args.member} done through epoch {end_epoch}.')
