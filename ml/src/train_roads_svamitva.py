"""
Train road model specifically on SVAMITVA Indian villages (Lalpur, Gujarat)
using the 4 real SVAMITVA images + road masks generated from ProjectVaayu ShapeFiles.

Dataset: 4 images 1000x1000 from /dashboard/public/svamitva/svamitva-*.jpg + _road_mask.png
Augmented to ~400 patches via sliding window + synthetic variations.

Also mixes in Massachusetts roads for generalization, but weighted toward SVAMITVA.

Usage: python train_roads_svamitva.py --epochs 10
"""
import os, sys, time, random, argparse, io
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from model import AttentionResUNet

torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.set_num_threads(4)

SVAM_DIR = "/home/rigalis/SKYMAP/dashboard/public/svamitva"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "road_model_svamitva.pt")
HIST_PATH = os.path.join(OUT_DIR, "road_history_svamitva.npy")
PARQUET = "/tmp/hf_cache/datasets--asolodin--massachusetts_roads_dataset_500/snapshots/5b203bfa4c5ee49308cd6205ceef3640bfe3e54f/data/validation-00000-of-00001.parquet"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()

device = torch.device("cpu")

class SVAMITVARoads(Dataset):
    def __init__(self, svam_dir, patch=128, augment=True):
        self.files = sorted([f for f in os.listdir(svam_dir) if f.startswith("svamitva-") and f.endswith(".jpg")])
        self.masks = {f.replace(".jpg","_road_mask.png"): os.path.join(svam_dir, f.replace(".jpg","_road_mask.png")) for f in self.files}
        self.patch = patch
        self.augment = augment
        self.samples = []
        for fname in self.files:
            img_path = os.path.join(svam_dir, fname)
            mask_path = os.path.join(svam_dir, fname.replace(".jpg","_road_mask.png"))
            img = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L"))
            mask = (mask>127).astype(np.float32)
            H,W = mask.shape
            # need to create patches via sliding window
            stride = 64 if augment else 96
            for y in range(0, H-patch+1, stride):
                for x in range(0, W-patch+1, stride):
                    self.samples.append((img, mask, y, x))
        print(f"SVAMITVA {'train' if augment else 'val'}: {len(self.samples)} patches from {len(self.files)} images")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, mask, y, x = self.samples[idx]
        p=self.patch
        im = img[y:y+p, x:x+p].copy().astype(np.float32)/255.0
        ms = mask[y:y+p, x:x+p].copy()
        if self.augment:
            if random.random()<0.5: im=np.fliplr(im).copy(); ms=np.fliplr(ms).copy()
            if random.random()<0.5: im=np.flipud(im).copy(); ms=np.flipud(ms).copy()
            k=random.randint(0,3); im=np.rot90(im,k).copy(); ms=np.rot90(ms,k).copy()
            if random.random()<0.7:
                hsv=cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-4,4))%180
                hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(0.9,1.1),0,255)
                hsv[:,:,2]=np.clip(hsv[:,:,2]*random.uniform(0.95,1.05),0,255)
                im=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
        # resize not needed, already 128
        im_t=torch.from_numpy(im.transpose(2,0,1)).float()
        ms_t=torch.from_numpy(ms).float().unsqueeze(0)
        return im_t, ms_t

class MassRoadsSmall(Dataset):
    def __init__(self, parquet, augment=True):
        import pyarrow.parquet as pq
        import io
        from PIL import Image
        table=pq.read_table(parquet)
        rows=table.to_pylist()
        random.shuffle(rows)
        # use subset 80/20
        n=int(len(rows)*0.85)
        self.rows=rows[:n] if augment else rows[n:]
        self.augment=augment
        print(f"MassRoads {'train' if augment else 'val'}: {len(self.rows)} images")
    def __len__(self): return len(self.rows)*3
    def __getitem__(self, idx):
        import io, cv2
        row=self.rows[(idx//3)%len(self.rows)]
        img=Image.open(io.BytesIO(row['image']['bytes'])).convert("RGB")
        lbl=Image.open(io.BytesIO(row['label']['bytes'])).convert("L")
        W,H=img.size
        x=random.randint(0,W-128); y=random.randint(0,H-128)
        img=img.crop((x,y,x+128,y+128)); lbl=lbl.crop((x,y,x+128,y+128))
        im=np.array(img).astype(np.float32)/255.0
        ms=(np.array(lbl)>127).astype(np.float32)
        if self.augment:
            if random.random()<0.5: im=np.fliplr(im).copy(); ms=np.fliplr(ms).copy()
            if random.random()<0.5: im=np.flipud(im).copy(); ms=np.flipud(ms).copy()
            k=random.randint(0,3); im=np.rot90(im,k).copy(); ms=np.rot90(ms,k).copy()
        im_t=torch.from_numpy(im.transpose(2,0,1)).float()
        ms_t=torch.from_numpy(ms).float().unsqueeze(0)
        return im_t, ms_t

# Build datasets
sv_train = SVAMITVARoads(SVAM_DIR, patch=128, augment=True)
sv_val = SVAMITVARoads(SVAM_DIR, patch=128, augment=False)
# use small mass for regularization
mass_train = MassRoadsSmall(PARQUET, augment=True)
mass_val = MassRoadsSmall(PARQUET, augment=False)

# Mix: SVAMITVA heavy (Indian villages) + Mass
train_ds = ConcatDataset([sv_train, mass_train])
val_ds = ConcatDataset([sv_val, mass_val])

print(f"Train total {len(train_ds)}, Val total {len(val_ds)}")
train_loader=DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
val_loader=DataLoader(val_ds, batch_size=args.batch)

model=AttentionResUNet(in_ch=3, out_ch=1, base=16).to(device)
opt=torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=2)
bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(6.0))
def dice_loss(pred, target, eps=1e-6):
    pred=torch.sigmoid(pred)
    inter=(pred*target).sum(dim=(1,2,3))
    union=pred.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))
    return 1-((2*inter+eps)/(union+eps)).mean()
def combined(pred, target): return 0.5*bce(pred,target)+0.5*dice_loss(pred,target)
def iou(pred, target, eps=1e-6):
    pred=(torch.sigmoid(pred)>0.5).float()
    inter=(pred*target).sum(dim=(1,2,3))
    union=((pred+target)>0).float().sum(dim=(1,2,3))
    return ((inter+eps)/(union+eps)).mean().item()

best=-1; hist={'train_loss':[],'val_loss':[],'val_iou':[]}
t0=time.time()
for epoch in range(1, args.epochs+1):
    model.train(); tr=[]
    for im, ms in train_loader:
        im,ms=im.to(device),ms.to(device)
        opt.zero_grad()
        pred=model(im)
        loss=combined(pred,ms)
        loss.backward(); opt.step()
        tr.append(loss.item())
    model.eval(); va=[]; ious=[]
    with torch.no_grad():
        for im, ms in val_loader:
            im,ms=im.to(device),ms.to(device)
            pred=model(im)
            va.append(combined(pred,ms).item())
            ious.append(iou(pred,ms))
    tr_l=np.mean(tr); va_l=np.mean(va); va_iou=np.mean(ious)
    scheduler.step(va_l)
    hist['train_loss'].append(tr_l); hist['val_loss'].append(va_l); hist['val_iou'].append(va_iou)
    print(f"Epoch {epoch:2d}/{args.epochs} | train {tr_l:.4f} val {va_l:.4f} IoU {va_iou:.4f} | {time.time()-t0:.0f}s")
    if va_iou>best:
        best=va_iou
        torch.save({'model':model.state_dict(),'epoch':epoch,'iou':va_iou}, CKPT_PATH)
        print(f" -> saved {CKPT_PATH} IoU {va_iou:.4f}")
    np.save(HIST_PATH, hist)
print(f"Done best IoU {best:.4f} -> {CKPT_PATH}")
