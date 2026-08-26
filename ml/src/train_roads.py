"""
Train road segmentation model for Indian villages (SVAMITVA).

Uses:
  - Real road labels: Massachusetts Roads 500 validation split (126 images, 500x500)
    downloaded as parquet — real road geometry, then Indian color augmentation
    to simulate dust/haze of Indian villages (hue shift + brightness).
  - Synthetic Indian villages roads: generated on SVAMITVA RGB.png background
    with procedural road networks (Indian village pattern: narrow, curvy, dirt)
  Combined ~300+ patches for training AttentionResUNet.

Usage:
  python train_roads.py --epochs 8 --batch 8

Outputs:
  outputs/road_model_ckpt.pt  (best model)
  outputs/road_history.npy

Then inference via infer_upload.py / serve.py will auto-use this ckpt if present.
"""
import os, sys, time, argparse, io, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset

from model import AttentionResUNet

torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.set_num_threads(4)

# paths
PARQUET_PATH = "/tmp/hf_cache/datasets--asolodin--massachusetts_roads_dataset_500/snapshots/5b203bfa4c5ee49308cd6205ceef3640bfe3e54f/data/validation-00000-of-00001.parquet"
SVAMITVA_RGB = "/home/rigalis/SIH2027/svamitva_prototype/final_package/data/RGB.png"
# fallback to survey if not found
if not os.path.exists(SVAMITVA_RGB):
    SVAMITVA_RGB = "/home/rigalis/SKYMAP/dashboard/public/survey/area-01.jpg"

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "road_model_ckpt.pt")
HIST_PATH = os.path.join(OUT_DIR, "road_history.npy")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=8)
parser.add_argument("--batch", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

device = torch.device("cpu")

# ---- Dataset 1: Massachusetts Roads 500 (real labels) ----
class MassRoadsParquet(Dataset):
    def __init__(self, parquet_path, split_train=True, train_frac=0.85, augment=True):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        # shuffle
        random.shuffle(rows)
        n = len(rows)
        n_train = int(n*train_frac)
        self.rows = rows[:n_train] if split_train else rows[n_train:]
        self.augment = augment
        print(f"MassRoads {'train' if split_train else 'val'}: {len(self.rows)} images from {parquet_path}")

    def __len__(self): return len(self.rows)*4  # 4 patches per 500x500 image

    def __getitem__(self, idx):
        # pick image and random crop 256x256
        row_idx = (idx // 4) % len(self.rows)
        row = self.rows[row_idx]
        img = Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
        lbl = Image.open(io.BytesIO(row['label']['bytes'])).convert("L")
        # random 256 crop
        W,H = img.size
        x = random.randint(0, W-256)
        y = random.randint(0, H-256)
        img = img.crop((x,y,x+256,y+256))
        lbl = lbl.crop((x,y,x+256,y+256))
        img = np.array(img).astype(np.float32)/255.0
        lbl = (np.array(lbl) > 127).astype(np.float32)
        # Indian village color augmentation: shift towards dust/brown
        if self.augment and random.random()<0.7:
            # hue shift in HSV: roads in India more brown/dust
            hsv = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-5, 8)) % 180
            hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.85,1.15), 0,255)
            hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.9,1.1), 0,255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
            # add haze (slight desaturation + brightness)
            if random.random()<0.3:
                img = np.clip(img*0.92 + 0.08, 0,1)
        # flips/rot
        if self.augment:
            if random.random()<0.5: img=np.fliplr(img).copy(); lbl=np.fliplr(lbl).copy()
            if random.random()<0.5: img=np.flipud(img).copy(); lbl=np.flipud(lbl).copy()
            k=random.randint(0,3); img=np.rot90(img,k).copy(); lbl=np.rot90(lbl,k).copy()
        # resize to 128 for model
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (128,128), interpolation=cv2.INTER_NEAREST)
        img_t = torch.from_numpy(img.transpose(2,0,1)).float()
        lbl_t = torch.from_numpy(lbl).float().unsqueeze(0)
        return img_t, lbl_t

# ---- Dataset 2: Synthetic Indian village roads on SVAMITVA background ----
class SyntheticIndianRoads(Dataset):
    def __init__(self, bg_path, num_samples=200, augment=True):
        self.bg = np.array(Image.open(bg_path).convert("RGB")).astype(np.float32)/255.0
        self.H,self.W,_ = self.bg.shape
        self.num_samples = num_samples
        self.augment = augment
        print(f"Synthetic Indian roads: {num_samples} samples from {bg_path} {self.W}x{self.H}")

    def __len__(self): return self.num_samples

    def gen_mask(self, h=256,w=256):
        mask = np.zeros((h,w), np.uint8)
        n_roads = random.randint(1,4)
        for _ in range(n_roads):
            # Indian village: narrow (2-6px at 256 scale), curvy, dirt-like
            thickness = random.randint(2,6)
            # random polyline with curve
            pts = []
            x,y = random.randint(0,w-1), random.randint(0,h-1)
            pts.append((x,y))
            for _ in range(random.randint(2,5)):
                x = np.clip(x + random.randint(-80,80), 0, w-1)
                y = np.clip(y + random.randint(-80,80), 0, h-1)
                pts.append((x,y))
            color = 255
            # draw with slight wavy
            if len(pts)>=2:
                # smooth via approx
                pts_arr = np.array(pts, np.int32)
                cv2.polylines(mask, [pts_arr], False, color, thickness, lineType=cv2.LINE_AA)
                # add small gaps to simulate fragmented village roads
                if random.random()<0.3:
                    # erase random segment
                    cx,cy = random.randint(0,w-1), random.randint(0,h-1)
                    cv2.circle(mask, (cx,cy), random.randint(5,15), 0, -1)
        # also add some intersections
        return (mask>127).astype(np.float32)

    def __getitem__(self, idx):
        # random crop from bg
        x = random.randint(0, self.W-256)
        y = random.randint(0, self.H-256)
        bg_patch = self.bg[y:y+256, x:x+256].copy()
        mask = self.gen_mask(256,256)
        # composite road color onto bg: dirt brown/gray
        road_color = random.choice([
            np.array([0.55,0.52,0.48]), # asphalt gray
            np.array([0.62,0.55,0.43]), # dirt brown
            np.array([0.78,0.78,0.75]), # concrete light
        ])
        # vary color slightly
        road_color = np.clip(road_color + np.random.uniform(-0.05,0.05,3), 0,1)
        # where mask, blend road color with bg
        rgb = bg_patch.copy()
        m3 = mask[:,:,None]
        # slight texture: add noise
        noise = np.random.uniform(-0.02,0.02, rgb.shape)
        rgb = np.clip(np.where(m3>0, road_color*0.85 + bg_patch*0.15 + noise*0.5, bg_patch),0,1)
        img = rgb
        lbl = mask
        if self.augment:
            if random.random()<0.5: img=np.fliplr(img).copy(); lbl=np.fliplr(lbl).copy()
            if random.random()<0.5: img=np.flipud(img).copy(); lbl=np.flipud(lbl).copy()
            k=random.randint(0,3); img=np.rot90(img,k).copy(); lbl=np.rot90(lbl,k).copy()
        img = cv2.resize(img, (128,128), interpolation=cv2.INTER_LINEAR)
        lbl = cv2.resize(lbl, (128,128), interpolation=cv2.INTER_NEAREST)
        img = (lbl>0.5).astype(np.float32) if False else img # keep img
        img_t = torch.from_numpy(img.transpose(2,0,1)).float()
        lbl_t = torch.from_numpy(lbl).float().unsqueeze(0)
        return img_t, lbl_t

# build datasets
if not os.path.exists(PARQUET_PATH):
    print(f"Missing {PARQUET_PATH}, using synthetic only")
    train_ds = SyntheticIndianRoads(SVAMITVA_RGB, num_samples=400, augment=True)
    val_ds = SyntheticIndianRoads(SVAMITVA_RGB, num_samples=60, augment=False)
else:
    mass_train = MassRoadsParquet(PARQUET_PATH, split_train=True, augment=True)
    mass_val = MassRoadsParquet(PARQUET_PATH, split_train=False, augment=False)
    synth_train = SyntheticIndianRoads(SVAMITVA_RGB, num_samples=150, augment=True)
    synth_val = SyntheticIndianRoads(SVAMITVA_RGB, num_samples=30, augment=False)
    train_ds = ConcatDataset([mass_train, synth_train])
    val_ds = ConcatDataset([mass_val, synth_val])

print(f"Train: {len(train_ds)} patches, Val: {len(val_ds)} patches")
train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

# model
model = AttentionResUNet(in_ch=3, out_ch=1, base=16).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)

# class imbalance: roads are ~5-10% pixels
pos_weight = torch.tensor(8.0)  # cap
bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred*target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    return 1 - ((2*inter+eps)/(union+eps)).mean()

def combined_loss(pred, target):
    return 0.5*bce(pred,target) + 0.5*dice_loss(pred,target)

def iou_score(pred, target, thresh=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred)>thresh).float()
    inter = (pred*target).sum(dim=(1,2,3))
    union = ((pred+target)>0).float().sum(dim=(1,2,3))
    return ((inter+eps)/(union+eps)).mean().item()

history={'train_loss':[],'val_loss':[],'val_iou':[],'lr':[]}
best_iou=-1
t0=time.time()
for epoch in range(1, args.epochs+1):
    model.train()
    tr_losses=[]
    for img, msk in train_loader:
        img, msk = img.to(device), msk.to(device)
        opt.zero_grad()
        pred = model(img)
        loss = combined_loss(pred, msk)
        loss.backward()
        opt.step()
        tr_losses.append(loss.item())
    model.eval()
    val_losses, val_ious=[],[]
    with torch.no_grad():
        for img, msk in val_loader:
            img, msk = img.to(device), msk.to(device)
            pred = model(img)
            val_losses.append(combined_loss(pred,msk).item())
            val_ious.append(iou_score(pred,msk))
    tr_l, va_l, va_iou = np.mean(tr_losses), np.mean(val_losses), np.mean(val_ious)
    scheduler.step(va_l)
    cur_lr = opt.param_groups[0]['lr']
    history['train_loss'].append(tr_l); history['val_loss'].append(va_l); history['val_iou'].append(va_iou); history['lr'].append(cur_lr)
    print(f"Epoch {epoch:2d}/{args.epochs} | train {tr_l:.4f} | val {va_l:.4f} | IoU {va_iou:.4f} | lr {cur_lr:.1e} | {time.time()-t0:.0f}s")
    # save best
    if va_iou>best_iou:
        best_iou=va_iou
        torch.save({'model':model.state_dict(),'epoch':epoch,'iou':va_iou}, CKPT_PATH)
        print(f"  -> saved best {CKPT_PATH} IoU {va_iou:.4f}")
    np.save(HIST_PATH, history)

print(f"Done. Best IoU {best_iou:.4f} -> {CKPT_PATH}")
