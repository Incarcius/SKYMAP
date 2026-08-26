"""
Multi-scale fine-tuning of the main model.

Starts from the fully-converged epoch-27 checkpoint and continues training
with MultiScalePatchDataset (patches sampled at 96/128/160/192px, resized
to the fixed 128x128 network input). This is fine-tuning, not training from
scratch - the goal is to nudge the already-good feature representations
toward scale invariance without losing what was already learned.

Validation stays at the ORIGINAL fixed 128px scale (no resizing) so the
before/after comparison is apples-to-apples with every other number
reported for this model.
"""
import sys, time, argparse, os
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from PIL import Image

from model import AttentionResUNet
from dataset import PatchDataset, MultiScalePatchDataset, split_regions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
CKPT_PATH = f'{OUT_DIR}/model_multiscale_ckpt.pt'
SOURCE_CKPT = f'{OUT_DIR}/model_v2_ckpt.pt'  # epoch-27 single-scale model

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--resume', action='store_true')
parser.add_argument('--lr', type=float, default=3e-4)  # lower LR for fine-tuning, not training from scratch
args = parser.parse_args()

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(4)

rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
gt = np.array(Image.open(f'{DATA_DIR}/GT.png').convert('L')).astype(np.float32)
gt = (gt > 127).astype(np.float32)
H, W = gt.shape

train_region, val_region = split_regions(H, W, val_fraction=0.2, axis='x', side='right')

train_ds = MultiScalePatchDataset(rgb, gt, train_region, net_size=128,
                                   scales=(96, 128, 160, 192), stride_frac=0.45, augment=True)
val_ds = PatchDataset(rgb, gt, val_region, patch_size=128, stride=32, augment=False)  # fixed scale, comparable to main model
print(f'Multi-scale train patches: {len(train_ds)} (scales 96/128/160/192px)', flush=True)
print(f'Val patches (fixed 128px, for apples-to-apples comparison): {len(val_ds)}', flush=True)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

train_building_frac = gt[train_region[0]:train_region[1], train_region[2]:train_region[3]].mean()
pos_weight_val = min((1 - train_building_frac) / max(train_building_frac, 1e-4), 8.0)

model = AttentionResUNet(in_ch=3, out_ch=1, base=16)
start_epoch = 0
if args.resume and os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    start_epoch = ckpt['epoch']
    print(f'Resumed multi-scale fine-tuning from epoch {start_epoch}', flush=True)
else:
    source = torch.load(SOURCE_CKPT, map_location='cpu')
    model.load_state_dict(source['model'])
    print(f"Initialized from single-scale checkpoint (epoch {source['epoch']}), starting multi-scale fine-tuning", flush=True)

opt = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
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
    print(f'Epoch {epoch}/{end_epoch} | train_loss={tr_l:.4f} | val_loss={va_l:.4f} | '
          f'val_IoU(fixed-scale)={va_iou:.4f} | elapsed={time.time()-t0:.0f}s', flush=True)

    torch.save({'model': model.state_dict(), 'epoch': epoch}, CKPT_PATH)

print(f'Multi-scale fine-tuning done through epoch {end_epoch}.')
