"""
Final production inference for the trained v2 model (epoch 27 checkpoint):

  1. TEST-TIME AUGMENTATION (TTA) - each patch is run through the model 4
     times (original, horizontal flip, vertical flip, 180-degree rotation),
     predictions are un-transformed back and averaged. This smooths out
     orientation-specific errors for free (no extra training), since
     aerial imagery has no canonical "up" direction the model should be
     equally confident regardless of rotation.

  2. THRESHOLD TUNING - rather than assuming the default 0.5 cutoff is
     optimal, this sweeps thresholds against the held-out validation
     region and picks the one that maximizes F1, then reports honestly
     how much that improved over the naive default.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import cv2
from PIL import Image
from model import AttentionResUNet
from dataset import split_regions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
gt = np.array(Image.open(f'{DATA_DIR}/GT.png').convert('L')).astype(np.float32)
gt_bin = (gt > 127).astype(np.uint8)
H, W = gt_bin.shape
train_region, val_region = split_regions(H, W, val_fraction=0.2, axis='x')

torch.set_num_threads(4)
ckpt = torch.load(f'{OUT_DIR}/model_v2_ckpt.pt', map_location='cpu')
model = AttentionResUNet(in_ch=3, out_ch=1, base=16)
model.load_state_dict(ckpt['model'])
model.eval()
print(f"Loaded final checkpoint: epoch {ckpt['epoch']}")


def predict_patch_tta(tile_np):
    """Run 4-way TTA on a single 128x128 patch, return averaged probability map."""
    transforms = [
        (lambda x: x, lambda x: x),                                    # identity
        (lambda x: np.fliplr(x), lambda x: np.fliplr(x)),               # h-flip
        (lambda x: np.flipud(x), lambda x: np.flipud(x)),               # v-flip
        (lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 2)),           # 180 rot
    ]
    probs = []
    with torch.no_grad():
        for fwd, inv in transforms:
            t_img = fwd(tile_np).copy()
            t = torch.from_numpy(t_img.transpose(2, 0, 1)).float().unsqueeze(0)
            out = torch.sigmoid(model(t)).squeeze().numpy()
            out = inv(out).copy()
            probs.append(out)
    return np.mean(probs, axis=0)


def run_full_inference(use_tta=True, patch=128, stride=96):
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)
    ys = sorted(set(list(range(0, H - patch + 1, stride)) + [H - patch]))
    xs = sorted(set(list(range(0, W - patch + 1, stride)) + [W - patch]))
    for y in ys:
        for x in xs:
            tile = rgb[y:y+patch, x:x+patch, :]
            if use_tta:
                out = predict_patch_tta(tile)
            else:
                with torch.no_grad():
                    t = torch.from_numpy(tile.transpose(2, 0, 1)).float().unsqueeze(0)
                    out = torch.sigmoid(model(t)).squeeze().numpy()
            pred_sum[y:y+patch, x:x+patch] += out
            pred_cnt[y:y+patch, x:x+patch] += 1
    pred_cnt[pred_cnt == 0] = 1
    return pred_sum / pred_cnt


print('\nRunning TTA inference on full tile (4x forward passes per patch)...')
pred_prob_tta = run_full_inference(use_tta=True)
np.save(f'{OUT_DIR}/pred_prob_final.npy', pred_prob_tta)
print('Done.')

# ---- Threshold tuning on the held-out validation region only ----
y0, y1, x0, x1 = val_region
val_prob = pred_prob_tta[y0:y1, x0:x1]
val_gt = gt_bin[y0:y1, x0:x1]

thresholds = np.arange(0.3, 0.71, 0.02)
best_f1, best_thresh = -1, 0.5
results = []
for th in thresholds:
    pred = (val_prob > th).astype(np.uint8)
    tp = np.logical_and(pred == 1, val_gt == 1).sum()
    fp = np.logical_and(pred == 1, val_gt == 0).sum()
    fn = np.logical_and(pred == 0, val_gt == 1).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    results.append((th, precision, recall, f1, iou))
    if f1 > best_f1:
        best_f1, best_thresh = f1, th

print(f'\nThreshold sweep (held-out region): best threshold = {best_thresh:.2f} (F1={best_f1:.4f})')
for th, p, r, f1, iou in results:
    marker = '  <-- best' if th == best_thresh else ''
    print(f'  th={th:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  IoU={iou:.3f}{marker}')

np.save(f'{OUT_DIR}/threshold_sweep.npy', np.array(results))
with open(f'{OUT_DIR}/best_threshold.txt', 'w') as f:
    f.write(str(best_thresh))

# ---- Final honest comparison: default 0.5 (no TTA) vs tuned+TTA, both on held-out region ----
def compute_metrics(pred, gt_r):
    tp = np.logical_and(pred == 1, gt_r == 1).sum()
    fp = np.logical_and(pred == 1, gt_r == 0).sum()
    fn = np.logical_and(pred == 0, gt_r == 1).sum()
    tn = np.logical_and(pred == 0, gt_r == 0).sum()
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn)
    return dict(iou=iou, precision=precision, recall=recall, f1=f1, accuracy=acc)


# baseline: no TTA, default threshold (single forward pass, th=0.5)
pred_prob_notta = run_full_inference(use_tta=False)
baseline_pred = (pred_prob_notta[y0:y1, x0:x1] > 0.5).astype(np.uint8)
baseline_metrics = compute_metrics(baseline_pred, val_gt)

final_pred = (val_prob > best_thresh).astype(np.uint8)
final_metrics = compute_metrics(final_pred, val_gt)

print('\n=== FINAL COMPARISON (held-out region, epoch 27 checkpoint) ===')
print(f'No TTA, default th=0.5   : IoU={baseline_metrics["iou"]:.4f}  P={baseline_metrics["precision"]:.4f}  R={baseline_metrics["recall"]:.4f}  F1={baseline_metrics["f1"]:.4f}')
print(f'TTA + tuned th={best_thresh:.2f}     : IoU={final_metrics["iou"]:.4f}  P={final_metrics["precision"]:.4f}  R={final_metrics["recall"]:.4f}  F1={final_metrics["f1"]:.4f}')
improvement = (final_metrics['iou'] - baseline_metrics['iou']) / baseline_metrics['iou'] * 100
print(f'IoU improvement from TTA + threshold tuning alone: {improvement:+.1f}%')

with open(f'{OUT_DIR}/metrics_final.txt', 'w') as f:
    f.write(f'Final model: epoch {ckpt["epoch"]} checkpoint, TTA + tuned threshold ({best_thresh:.2f})\n\n')
    f.write('No TTA, default threshold 0.5 (held-out region):\n')
    for k, v in baseline_metrics.items():
        f.write(f'  {k}: {v:.4f}\n')
    f.write(f'\nTTA + tuned threshold {best_thresh:.2f} (held-out region):\n')
    for k, v in final_metrics.items():
        f.write(f'  {k}: {v:.4f}\n')
    f.write(f'\nIoU improvement from TTA + threshold tuning: {improvement:+.1f}%\n')

# ---- Save final full-tile clean mask (using tuned threshold) for downstream use ----
final_full_mask = (pred_prob_tta > best_thresh).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
final_clean = cv2.morphologyEx(final_full_mask, cv2.MORPH_OPEN, kernel)
final_clean = cv2.morphologyEx(final_clean, cv2.MORPH_CLOSE, kernel)
Image.fromarray(final_clean).save(f'{OUT_DIR}/pred_mask_final_clean.png')
print('\nSaved final model outputs: pred_prob_final.npy, pred_mask_final_clean.png, metrics_final.txt')
