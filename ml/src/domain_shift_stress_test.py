"""
Domain-shift stress test.

Every metric reported so far is on THIS tile, THIS lighting, THIS season.
A real deployment sees drone passes across different times of day, seasons,
and sensor calibrations. Rather than just asserting "the model generalizes"
without evidence, this applies synthetic transformations simulating those
shifts to the held-out region, re-runs the actual trained ensemble on each,
and reports how much IoU degrades under each condition - honest evidence
for (or against) the robustness claim, not just a marketing line.

Conditions tested:
  - baseline (no transform)
  - brighter (simulates high summer sun / overexposure)
  - darker (simulates overcast conditions)
  - low contrast + haze (simulates fog/atmospheric haze)
  - hue shift (simulates different sensor color calibration)
  - combined worst-case (darker + hazy + hue-shifted together)

Only runs inference on the held-out validation region (not the full tile)
to keep this fast - the metric only needs that region anyway.
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

rgb_full = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
gt = np.array(Image.open(f'{DATA_DIR}/GT.png').convert('L')).astype(np.float32)
gt_bin = (gt > 127).astype(np.uint8)
H, W = gt_bin.shape
train_region, val_region = split_regions(H, W, val_fraction=0.2, axis='x', side='right')
y0, y1, x0, x1 = val_region

rgb_crop = rgb_full[y0:y1, x0:x1].copy()
gt_crop = gt_bin[y0:y1, x0:x1]
crop_h, crop_w = rgb_crop.shape[:2]

torch.set_num_threads(4)
with open(f'{OUT_DIR}/ensemble_threshold.txt') as f:
    THRESH = float(f.read().strip())


def load_model(path):
    ckpt = torch.load(path, map_location='cpu')
    m = AttentionResUNet(in_ch=3, out_ch=1, base=16)
    m.load_state_dict(ckpt['model'])
    m.eval()
    return m


models = [load_model(f'{OUT_DIR}/model_v2_ckpt.pt'),
          load_model(f'{OUT_DIR}/ensemble_member_1.pt'),
          load_model(f'{OUT_DIR}/ensemble_member_2.pt')]


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
            probs.append(inv(out).copy())
    return np.mean(probs, axis=0)


def run_ensemble_inference(img, patch=128, stride=64):
    h, w = img.shape[:2]
    pred_sum = np.zeros((h, w), dtype=np.float32)
    pred_cnt = np.zeros((h, w), dtype=np.float32)
    ys = sorted(set(list(range(0, h - patch + 1, stride)) + [max(0, h - patch)]))
    xs = sorted(set(list(range(0, w - patch + 1, stride)) + [max(0, w - patch)]))
    for y in ys:
        for x in xs:
            tile = img[y:y + patch, x:x + patch, :]
            if tile.shape[0] != patch or tile.shape[1] != patch:
                continue
            member_probs = [predict_patch_tta(m, tile) for m in models]
            avg = np.mean(member_probs, axis=0)
            pred_sum[y:y + patch, x:x + patch] += avg
            pred_cnt[y:y + patch, x:x + patch] += 1
    pred_cnt[pred_cnt == 0] = 1
    return pred_sum / pred_cnt


def compute_iou(pred_bin, gt_bin_):
    tp = np.logical_and(pred_bin == 1, gt_bin_ == 1).sum()
    fp = np.logical_and(pred_bin == 1, gt_bin_ == 0).sum()
    fn = np.logical_and(pred_bin == 0, gt_bin_ == 1).sum()
    return tp / (tp + fp + fn + 1e-6)


def apply_brighten(img, amount=0.18):
    return np.clip(img + amount, 0, 1)


def apply_darken(img, amount=0.22):
    return np.clip(img - amount, 0, 1)


def apply_haze(img, strength=0.35):
    # blend toward flat gray + reduce contrast - simulates atmospheric haze/fog
    gray_fog = np.ones_like(img) * 0.75
    hazy = img * (1 - strength) + gray_fog * strength
    hazy = np.clip((hazy - 0.5) * 0.7 + 0.5, 0, 1)
    return hazy


def apply_hue_shift(img, deg=25):
    img_u8 = (img * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[:, :, 0] = (hsv[:, :, 0] + deg) % 180
    hsv = hsv.astype(np.uint8)
    shifted = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return shifted.astype(np.float32) / 255.0


def apply_combined_worst_case(img):
    return apply_hue_shift(apply_haze(apply_darken(img, 0.15), 0.3), 15)


conditions = [
    ('Baseline (no shift)', lambda x: x),
    ('Brighter (+overexposed)', apply_brighten),
    ('Darker (overcast)', apply_darken),
    ('Hazy/foggy (low contrast)', apply_haze),
    ('Hue-shifted (sensor drift)', apply_hue_shift),
    ('Combined worst-case', apply_combined_worst_case),
]

print(f'Running domain-shift stress test on held-out region ({crop_h}x{crop_w}px), '
      f'3-model ensemble + TTA, threshold={THRESH:.2f}\n')

results = []
baseline_iou = None
for name, fn in conditions:
    transformed = fn(rgb_crop)
    prob = run_ensemble_inference(transformed)
    pred_bin = (prob > THRESH).astype(np.uint8)
    iou = compute_iou(pred_bin, gt_crop)
    if baseline_iou is None:
        baseline_iou = iou
    delta_pct = (iou - baseline_iou) / baseline_iou * 100
    results.append({'condition': name, 'iou': float(iou), 'delta_pct': float(delta_pct)})
    print(f'{name:30s} IoU={iou:.4f}  ({delta_pct:+.1f}% vs baseline)')

    # save one example visualization per condition
    vis = (transformed * 255).astype(np.uint8)
    safe_name = ''.join(c for c in name.split('(')[0] if c.isalnum() or c == ' ').strip().replace(' ', '_').lower()
    Image.fromarray(vis).save(f'{OUT_DIR}/stress_test_{safe_name}.png')

import json
with open(f'{OUT_DIR}/domain_shift_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'\nSaved domain_shift_results.json and per-condition example images.')
