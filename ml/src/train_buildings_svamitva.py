"""
Train building model on SVAMITVA Indian villages (Lalpur) + original US tile.

Dataset:
 - SVAMITVA: 4 images 1000x1000 + bld_mask.png from ProjectVaayu ShapeFiles (Gujarat Lalpur)
   6-8% building pixels, 317 buildings total
 - Original: RGB.png + GT.png from SIH2027 prototype (US suburb, 9% buildings)

Model: AttentionResUNet base 16
Loss: BCE pos_weight + Dice
"""
import os, sys, time, random, argparse
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
ORIG_RGB = "/home/rigalis/SIH2027/svamitva_prototype/final_package/data/RGB.png"
ORIG_GT = "/home/rigalis/SIH2027/svamitva_prototype/final_package/data/GT.png"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT_PATH = os.path.join(OUT_DIR, "building_model_svamitva.pt")
HIST_PATH = os.path.join(OUT_DIR, "building_history_svamitva.npy")

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=12)
parser.add_argument("--batch", type=int, default=8)
args = parser.parse_args()
device=torch.device("cpu")

class SVAMITVABuilding(Dataset):
    def __init__(self, svam_dir, patch=128, augment=True):
        self.files = sorted([f for f in os.listdir(svam_dir) if f.startswith("svamitva-") and f.endswith(".jpg")])
        self.patch=patch
        self.augment=augment
        self.samples=[]
        for fname in self.files:
            img_path=os.path.join(svam_dir, fname)
            mask_path=os.path.join(svam_dir, fname.replace(".jpg","_bld_mask.png"))
            img=np.array(Image.open(img_path).convert("RGB"))
            mask=np.array(Image.open(mask_path).convert("L"))
            mask=(mask>127).astype(np.float32)
            H,W=mask.shape
            stride=64 if augment else 96
            for y in range(0, H-patch+1, stride):
                for x in range(0, W-patch+1, stride):
                    self.samples.append((img, mask, y, x))
        print(f"SVAMITVA {'train' if augment else 'val'}: {len(self.samples)} patches from {len(self.files)} images")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, mask, y, x = self.samples[idx]
        p=self.patch
        im=img[y:y+p, x:x+p].copy().astype(np.float32)/255.0
        ms=mask[y:y+p, x:x+p].copy()
        if self.augment:
            if random.random()<0.5: im=np.fliplr(im).copy(); ms=np.fliplr(ms).copy()
            if random.random()<0.5: im=np.flipud(im).copy(); ms=np.flipud(ms).copy()
            k=random.randint(0,3); im=np.rot90(im,k).copy(); ms=np.rot90(ms,k).copy()
            if random.random()<0.7:
                hsv=cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-6,6))%180
                hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(0.9,1.1),0,255)
                hsv[:,:,2]=np.clip(hsv[:,:,2]*random.uniform(0.9,1.1),0,255)
                im=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
        im_t=torch.from_numpy(im.transpose(2,0,1)).float()
        ms_t=torch.from_numpy(ms).float().unsqueeze(0)
        return im_t, ms_t

class OrigBuilding(Dataset):
    def __init__(self, rgb_path, gt_path, patch=128, augment=True, val_fraction=0.2):
        rgb=np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)/255.0
        gt=np.array(Image.open(gt_path).convert("L")).astype(np.float32)
        gt=(gt>127).astype(np.float32)
        H,W=gt.shape
        # spatial split for train/val like dataset.py
        split=int(W*(1-val_fraction))
        # use same logic as PatchDataset but we need to know if this is train or val
        # For simplicity, OrigBuilding will be used as train with left 80% and val with right 20% via separate instances
        # We'll handle via region param
        self.rgb=rgb; self.mask=gt; self.patch=patch; self.augment=augment
        self.H=H; self.W=W
        self.samples=[]
        # region will be set by caller via manual split; for now we create all patches and filter by x
        # This class will be instantiated twice with different regions
        # To keep simple, we create patches for whole image but filter
        # Instead we accept region as param in __init__
        raise NotImplementedError

class OrigBuildingRegion(Dataset):
    def __init__(self, rgb_path, gt_path, region, patch=128, augment=True):
        rgb=np.array(Image.open(rgb_path).convert("RGB")).astype(np.float32)/255.0
        gt=np.array(Image.open(gt_path).convert("L")).astype(np.float32)
        gt=(gt>127).astype(np.float32)
        self.rgb=rgb; self.mask=gt; self.patch=patch; self.augment=augment
        y0,y1,x0,x1=region
        self.samples=[]
        stride=48 if augment else 64
        for y in range(y0, y1-patch+1, stride):
            for x in range(x0, x1-patch+1, stride):
                self.samples.append((y,x))
        print(f"Orig {'train' if augment else 'val'} region {region}: {len(self.samples)} patches")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        y,x=self.samples[idx]
        p=self.patch
        im=self.rgb[y:y+p, x:x+p].copy()
        ms=self.mask[y:y+p, x:x+p].copy()
        if self.augment:
            if random.random()<0.5: im=np.fliplr(im).copy(); ms=np.fliplr(ms).copy()
            if random.random()<0.5: im=np.flipud(im).copy(); ms=np.flipud(ms).copy()
            k=random.randint(0,3); im=np.rot90(im,k).copy(); ms=np.rot90(ms,k).copy()
            if random.random()<0.7:
                hsv=cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-6,6))%180
                hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(0.9,1.1),0,255)
                hsv[:,:,2]=np.clip(hsv[:,:,2]*random.uniform(0.9,1.1),0,255)
                im=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
        im_t=torch.from_numpy(im.transpose(2,0,1)).float()
        ms_t=torch.from_numpy(ms).float().unsqueeze(0)
        return im_t, ms_t

# Build datasets
# Orig split 80% train (left) 20% val (right) as in train.py
from dataset import split_regions
import PIL.Image as PILImage
orig_rgb = np.array(Image.open(ORIG_RGB).convert("RGB"))
H,W,_ = orig_rgb.shape
train_region, val_region = split_regions(H,W, val_fraction=0.2, axis='x')
print(f"Orig train {train_region} val {val_region}")

sv_train = SVAMITVABuilding(SVAM_DIR, patch=128, augment=True)
sv_val = SVAMITVABuilding(SVAM_DIR, patch=128, augment=False)
orig_train = OrigBuildingRegion(ORIG_RGB, ORIG_GT, train_region, patch=128, augment=True)
orig_val = OrigBuildingRegion(ORIG_RGB, ORIG_GT, val_region, patch=128, augment=False)

train_ds = ConcatDataset([sv_train, orig_train])
val_ds = ConcatDataset([sv_val, orig_val])
print(f"Train total {len(train_ds)}, Val total {len(val_ds)}")
train_loader=DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
val_loader=DataLoader(val_ds, batch_size=args.batch)

model=AttentionResUNet(in_ch=3, out_ch=1, base=16).to(device)
# try load previous best if exists to continue
if os.path.exists(CKPT_PATH):
    try:
        ckpt=torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt['model'])
        print(f"Resumed from {CKPT_PATH} epoch {ckpt.get('epoch')}")
    except: pass
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
