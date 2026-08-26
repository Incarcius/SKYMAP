"""
Train road model JUST for SVAMITVA village images (4 real satellite + shapefile).
No Massachusetts — pure Indian villages, to fix 'very bad' on villages.
"""
import os, sys, time, random, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import AttentionResUNet

torch.manual_seed(42); np.random.seed(42); random.seed(42)
torch.set_num_threads(4)

SVAM_DIR = "/home/rigalis/SKYMAP/dashboard/public/svamitva"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)
CKPT = os.path.join(OUT_DIR, "road_model_village_only.pt")
HIST = os.path.join(OUT_DIR, "road_history_village_only.npy")

parser=argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--batch", type=int, default=8)
args=parser.parse_args()
device=torch.device("cpu")

class VillageRoads(Dataset):
    def __init__(self, svam_dir, patch=128, train=True):
        files=sorted([f for f in os.listdir(svam_dir) if f.startswith("svamitva-") and f.endswith(".jpg")])
        # 3 train, 1 val split (village 04 as val)
        if train:
            files=files[:3]  # 01,02,03 train
        else:
            files=files[3:]  # 04 val
        self.files=files
        self.patch=patch
        self.train=train
        self.samples=[]
        for fname in files:
            img=np.array(Image.open(os.path.join(svam_dir, fname)).convert("RGB"))
            mask=np.array(Image.open(os.path.join(svam_dir, fname.replace(".jpg","_road_mask.png"))).convert("L"))
            mask=(mask>127).astype(np.float32)
            H,W=mask.shape
            stride=48 if train else 64
            for y in range(0, H-patch+1, stride):
                for x in range(0, W-patch+1, stride):
                    self.samples.append((img, mask, y, x))
        print(f"Village {'train' if train else 'val'}: {len(self.samples)} patches from {files}")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img, mask, y, x = self.samples[idx]
        p=self.patch
        im=img[y:y+p, x:x+p].copy().astype(np.float32)/255.0
        ms=mask[y:y+p, x:x+p].copy()
        if self.train:
            if random.random()<0.5: im=np.fliplr(im).copy(); ms=np.fliplr(ms).copy()
            if random.random()<0.5: im=np.flipud(im).copy(); ms=np.flipud(ms).copy()
            k=random.randint(0,3); im=np.rot90(im,k).copy(); ms=np.rot90(ms,k).copy()
            # color jitter for Indian village satellite
            if random.random()<0.8:
                hsv=cv2.cvtColor((im*255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:,:,0]=(hsv[:,:,0]+random.uniform(-5,5))%180
                hsv[:,:,1]=np.clip(hsv[:,:,1]*random.uniform(0.92,1.08),0,255)
                hsv[:,:,2]=np.clip(hsv[:,:,2]*random.uniform(0.92,1.08),0,255)
                im=cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)/255.0
            # random brightness
            if random.random()<0.3:
                im=np.clip(im*random.uniform(0.9,1.1)+random.uniform(-0.02,0.02),0,1)
        im_t=torch.from_numpy(im.transpose(2,0,1)).float()
        ms_t=torch.from_numpy(ms).float().unsqueeze(0)
        return im_t, ms_t

train_ds=VillageRoads(SVAM_DIR, patch=128, train=True)
val_ds=VillageRoads(SVAM_DIR, patch=128, train=False)
print(f"Train {len(train_ds)}, Val {len(val_ds)}")
train_loader=DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
val_loader=DataLoader(val_ds, batch_size=args.batch)

model=AttentionResUNet(in_ch=3, out_ch=1, base=16).to(device)
opt=torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
bce=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0))
def dice(pred, target, eps=1e-6):
    pred=torch.sigmoid(pred)
    inter=(pred*target).sum(dim=(1,2,3))
    union=pred.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))
    return 1-((2*inter+eps)/(union+eps)).mean()
def combined(pred,target): return 0.5*bce(pred,target)+0.5*dice(pred,target)
def iou(pred,target,eps=1e-6):
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
    print(f"Epoch {epoch:2d}/{args.epochs} | train {tr_l:.4f} val {va_l:.4f} IoU {va_iou:.4f} lr {opt.param_groups[0]['lr']:.1e} | {time.time()-t0:.0f}s")
    if va_iou>best:
        best=va_iou
        torch.save({'model':model.state_dict(),'epoch':epoch,'iou':va_iou}, CKPT)
        print(f" -> saved {CKPT} IoU {va_iou:.4f}")
    np.save(HIST, hist)
print(f"Done best {best:.4f} -> {CKPT}")

# also test on all 4 villages full image
print("\nFull image test:")
import torch.serialization
try:
    import numpy._core.multiarray
    torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
except: pass
ckpt=torch.load(CKPT, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt['model'])
model.eval()
for i in range(1,5):
    rgb=np.array(Image.open(f"{SVAM_DIR}/svamitva-{i:02d}.jpg").convert("RGB"))
    gt=np.array(Image.open(f"{SVAM_DIR}/svamitva-{i:02d}_road_mask.png").convert("L"))
    gt=(gt>127).astype(np.uint8)
    H,W=rgb.shape[:2]
    rgb_f=rgb.astype(np.float32)/255.0
    patch=128; stride=96
    pred_sum=np.zeros((H,W),np.float32)
    pred_cnt=np.zeros((H,W),np.float32)
    ys=sorted(set(list(range(0,H-patch+1,stride))+[H-patch]))
    xs=sorted(set(list(range(0,W-patch+1,stride))+[W-patch]))
    import torch as th
    with th.no_grad():
        for y in ys:
            for x in xs:
                tile=rgb_f[y:y+patch,x:x+patch]
                t=th.from_numpy(tile.transpose(2,0,1)).float().unsqueeze(0)
                out=th.sigmoid(model(t)).squeeze().numpy()
                pred_sum[y:y+patch,x:x+patch]+=out
                pred_cnt[y:y+patch,x:x+patch]+=1
    pred_cnt[pred_cnt==0]=1
    prob=pred_sum/pred_cnt
    pred=(prob>0.5).astype(np.uint8)
    tp=((pred==1)&(gt==1)).sum(); fp=((pred==1)&(gt==0)).sum(); fn=((pred==0)&(gt==1)).sum()
    iou=tp/(tp+fp+fn+1e-6)
    print(f" svamitva-{i:02d} IoU {iou:.3f} prec {tp/(tp+fp+1e-6):.3f} rec {tp/(tp+fn+1e-6):.3f} cov pred {(pred>0).mean()*100:.2f}% gt {(gt>0).mean()*100:.2f}%")
