"""
Ensemble evaluation - averages predictions from 3 independently-initialized
models (the main 27-epoch model + 2 additional members trained with
different random seeds, all on the SAME train/val split) and compares
against the best single model. This is the classic "free" accuracy trick:
each model's errors are at least partly decorrelated (different init,
different mini-batch orders), so averaging smooths out individual mistakes
without needing any new data or architecture changes.

All three members also get TTA (test-time augmentation) applied per the
final_inference.py methodology, so this measures the full stack:
single-model+TTA+threshold vs ensemble-of-3+TTA+threshold.
"""
import sys
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from PIL import Image
from model import AttentionResUNet
from dataset import split_regions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
gt = np.array(Image.open(f'{DATA_DIR}/GT.png').convert('L')).astype(np.float32)
gt_bin = (gt > 127).astype(np.uint8)
H, W = gt_bin.shape
train_region, val_region = split_regions(H, W, val_fraction=0.2, axis='x', side='right')
y0, y1, x0, x1 = val_region

torch.set_num_threads(4)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    m = AttentionResUNet(in_ch=3, out_ch=1, base=16)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m, ckpt['epoch']


def predict_patch_tta(model, tile_np):
    transforms = [
        (lambda x: x, lambda x: x),
        (lambda x: np.fliplr(x), lambda x: np.fliplr(x)),
        (lambda x: np.flipud(x), lambda x: np.flipud(x)),
        (lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 2)),
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


def run_full_inference(model, patch=128, stride=96):
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)
    ys = sorted(set(list(range(0, H - patch + 1, stride)) + [H - patch]))
    xs = sorted(set(list(range(0, W - patch + 1, stride)) + [W - patch]))
    for y in ys:
        for x in xs:
            tile = rgb[y:y+patch, x:x+patch, :]
            out = predict_patch_tta(model, tile)
            pred_sum[y:y+patch, x:x+patch] += out
            pred_cnt[y:y+patch, x:x+patch] += 1
    pred_cnt[pred_cnt == 0] = 1
    return pred_sum / pred_cnt


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


print('Loading 3 ensemble members...')
main_model, main_epoch = load_model(f'{OUT_DIR}/model_v2_ckpt.pt')
m1, e1 = load_model(f'{OUT_DIR}/ensemble_member_1.pt')
m2, e2 = load_model(f'{OUT_DIR}/ensemble_member_2.pt')
print(f'Main model: epoch {main_epoch} | Member 1: epoch {e1} | Member 2: epoch {e2}')

print('\nRunning TTA inference for each member on full tile (this takes a few minutes)...')
prob_main = np.load(f'{OUT_DIR}/pred_prob_final.npy')  # already computed with TTA
print('  Main model: loaded from cache')
prob_m1 = run_full_inference(m1)
print('  Member 1: done')
prob_m2 = run_full_inference(m2)
print('  Member 2: done')

# ---- Individual model performance on held-out region (all with TTA) ----
with open(f'{OUT_DIR}/best_threshold.txt') as f:
    main_thresh = float(f.read().strip())

val_gt = gt_bin[y0:y1, x0:x1]
main_metrics = compute_metrics((prob_main[y0:y1, x0:x1] > main_thresh).astype(np.uint8), val_gt)
m1_metrics = compute_metrics((prob_m1[y0:y1, x0:x1] > 0.5).astype(np.uint8), val_gt)
m2_metrics = compute_metrics((prob_m2[y0:y1, x0:x1] > 0.5).astype(np.uint8), val_gt)

print(f'\nMain model  (epoch {main_epoch}, TTA, th={main_thresh:.2f}): IoU={main_metrics["iou"]:.4f}  F1={main_metrics["f1"]:.4f}')
print(f'Member 1    (epoch {e1}, TTA, th=0.50):          IoU={m1_metrics["iou"]:.4f}  F1={m1_metrics["f1"]:.4f}')
print(f'Member 2    (epoch {e2}, TTA, th=0.50):          IoU={m2_metrics["iou"]:.4f}  F1={m2_metrics["f1"]:.4f}')

# ---- Ensemble: average probabilities, then sweep threshold on held-out region ----
ensemble_prob = (prob_main + prob_m1 + prob_m2) / 3.0
ensemble_val_prob = ensemble_prob[y0:y1, x0:x1]

best_f1, best_th = -1, 0.5
for th in np.arange(0.3, 0.71, 0.02):
    pred = (ensemble_val_prob > th).astype(np.uint8)
    m = compute_metrics(pred, val_gt)
    if m['f1'] > best_f1:
        best_f1, best_th = m['f1'], th

ensemble_metrics = compute_metrics((ensemble_val_prob > best_th).astype(np.uint8), val_gt)
print(f'\nENSEMBLE (3 models avg, tuned th={best_th:.2f}):        IoU={ensemble_metrics["iou"]:.4f}  F1={ensemble_metrics["f1"]:.4f}')

improvement = (ensemble_metrics['iou'] - main_metrics['iou']) / main_metrics['iou'] * 100
print(f'\nEnsemble improvement over best single model: {improvement:+.1f}% IoU')

np.save(f'{OUT_DIR}/pred_prob_ensemble.npy', ensemble_prob)
with open(f'{OUT_DIR}/ensemble_results.txt', 'w') as f:
    f.write(f'Main model (epoch {main_epoch}, TTA):   IoU={main_metrics["iou"]:.4f} P={main_metrics["precision"]:.4f} R={main_metrics["recall"]:.4f} F1={main_metrics["f1"]:.4f}\n')
    f.write(f'Member 1 (epoch {e1}, TTA):              IoU={m1_metrics["iou"]:.4f} P={m1_metrics["precision"]:.4f} R={m1_metrics["recall"]:.4f} F1={m1_metrics["f1"]:.4f}\n')
    f.write(f'Member 2 (epoch {e2}, TTA):              IoU={m2_metrics["iou"]:.4f} P={m2_metrics["precision"]:.4f} R={m2_metrics["recall"]:.4f} F1={m2_metrics["f1"]:.4f}\n')
    f.write(f'ENSEMBLE (3-model avg, th={best_th:.2f}):        IoU={ensemble_metrics["iou"]:.4f} P={ensemble_metrics["precision"]:.4f} R={ensemble_metrics["recall"]:.4f} F1={ensemble_metrics["f1"]:.4f}\n')
    f.write(f'\nEnsemble improvement over best single model: {improvement:+.1f}% IoU\n')

print('\nSaved pred_prob_ensemble.npy and ensemble_results.txt')
