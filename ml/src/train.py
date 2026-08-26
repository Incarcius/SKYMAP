"""
Training script v2 - fixes over the baseline prototype:

  1. SPATIAL train/val split (split_regions in dataset.py) instead of a
     random split of overlapping sliding-window patches. The baseline
     version leaked pixels between train and val because neighbouring
     patches overlap heavily -> this version reserves a contiguous,
     completely untouched strip of the tile for validation.
  2. Class-imbalance handling via pos_weight in BCE (buildings are only
     ~9% of pixels; unweighted BCE biases the model toward predicting
     "non-building" everywhere).
  3. ReduceLROnPlateau LR scheduling.
  4. Resumable checkpointing - can be invoked repeatedly, each call trains
     N additional epochs and appends to the running history, so total
     training time isn't bound by a single process's wall-clock limit.

Usage:
  python3 train.py --epochs 6          # first call: trains 6 epochs from scratch
  python3 train.py --epochs 6 --resume # subsequent calls: resumes + trains 6 more
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

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
CKPT_PATH = f'{OUT_DIR}/model_v2_ckpt.pt'
HISTORY_PATH = f'{OUT_DIR}/history_v2.npy'

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=6, help='epochs to train THIS call')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
gt = np.array(Image.open(f'{DATA_DIR}/GT.png').convert('L')).astype(np.float32)
gt = (gt > 127).astype(np.float32)
H, W = gt.shape

train_region, val_region = split_regions(H, W, val_fraction=0.2, axis='x')
print(f'Train region (y0,y1,x0,x1): {train_region}  |  Val region: {val_region}', flush=True)

train_ds = PatchDataset(rgb, gt, train_region, patch_size=128, stride=48, augment=True)
val_ds = PatchDataset(rgb, gt, val_region, patch_size=128, stride=32, augment=False)  # denser sampling within the held-out region for a more stable metric (still zero overlap with train region)
print(f'Train patches: {len(train_ds)} (region size {train_region[1]-train_region[0]}x{train_region[3]-train_region[2]})', flush=True)
print(f'Val patches:   {len(val_ds)} (region size {val_region[1]-val_region[0]}x{val_region[3]-val_region[2]}) - SPATIALLY DISJOINT from train', flush=True)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# ---- Class imbalance: compute pos_weight from training region only ----
train_building_frac = gt[train_region[0]:train_region[1], train_region[2]:train_region[3]].mean()
pos_weight_val = (1 - train_building_frac) / max(train_building_frac, 1e-4)
pos_weight_val = min(pos_weight_val, 8.0)  # cap to avoid destabilizing training
print(f'Training-region building pixel fraction: {train_building_frac:.4f} -> BCE pos_weight={pos_weight_val:.2f}', flush=True)

device = torch.device('cpu')
model = AttentionResUNet(in_ch=3, out_ch=1, base=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

start_epoch = 0
history = {'train_loss': [], 'val_loss': [], 'val_iou': [], 'lr': []}

if args.resume and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    opt.load_state_dict(ckpt['opt'])
    start_epoch = ckpt['epoch']
    if os.path.exists(HISTORY_PATH):
        history = np.load(HISTORY_PATH, allow_pickle=True).item()
    print(f'Resumed from checkpoint at epoch {start_epoch}', flush=True)
else:
    print('Training from scratch', flush=True)

bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_val))


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
    cur_lr = opt.param_groups[0]['lr']

    history['train_loss'].append(tr_l)
    history['val_loss'].append(va_l)
    history['val_iou'].append(va_iou)
    history['lr'].append(cur_lr)

    print(f'Epoch {epoch:2d}/{end_epoch} | train_loss={tr_l:.4f} | val_loss={va_l:.4f} | '
          f'val_IoU={va_iou:.4f} | lr={cur_lr:.2e} | elapsed={time.time()-t0:.0f}s', flush=True)

    torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'epoch': epoch}, CKPT_PATH)
    np.save(HISTORY_PATH, history)

print(f'Done. Trained through epoch {end_epoch}. Checkpoint + history saved.', flush=True)
