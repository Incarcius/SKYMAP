"""
Change detection prototype.

HONESTY NOTE (put this on the slide, don't hide it): this sandbox has no
access to a real bi-temporal SVAMITVA orthophoto pair (two drone passes of
the same village, months apart). To demonstrate the *pipeline mechanics*
that real change detection would use, this script synthetically edits the
real INRIA tile to create a plausible "Time B" version: it paints in two
new rooftop-like structures (simulating unauthorized new construction) and
inpaints over one real existing building (simulating demolition). The
trained v2 model then runs independently on both tiles, and detections are
matched by IoU to flag what appeared / disappeared between passes. In
production this exact diff logic runs unchanged on two genuine dated
orthophotos - only the synthetic-edit step here is a stand-in for real
bi-temporal data, which the team does not have access to in this sandbox.
"""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import cv2
from PIL import Image
from model import AttentionResUNet

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
GSD_M = 1.5

rgb_a = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB'))
H, W = rgb_a.shape[:2]

# =====================================================================
# 1. Synthesize "Time B" tile: add 2 new structures, remove 1 existing one
# =====================================================================
rgb_b = rgb_a.copy()
rng = np.random.RandomState(7)

# --- simulate demolition: pick the detected building closest to a fixed
# reference location (robust to detection-count/ID drift between model
# versions, unlike hardcoding a specific building ID) ---
with open(f'{OUT_DIR}/buildings_final.json') as f:
    buildings_a_ref = json.load(f)
REFERENCE_XY = (623, 434)  # center of the building used in the original v2 demo
removed_building = min(
    buildings_a_ref,
    key=lambda b: (b['bbox'][0] + b['bbox'][2] / 2 - REFERENCE_XY[0]) ** 2
                  + (b['bbox'][1] + b['bbox'][3] / 2 - REFERENCE_XY[1]) ** 2
)
x, y, w, h = removed_building['bbox']
pad = 4
inpaint_mask = np.zeros((H, W), dtype=np.uint8)
inpaint_mask[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad] = 255
rgb_b_bgr = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2BGR)
rgb_b_bgr = cv2.inpaint(rgb_b_bgr, inpaint_mask, 7, cv2.INPAINT_TELEA)
rgb_b = cv2.cvtColor(rgb_b_bgr, cv2.COLOR_BGR2RGB)
print(f'Simulated demolition: removed building near reference point at bbox={removed_building["bbox"]}')


def paint_synthetic_building(img, cx, cy, w, h, base_color=(190, 188, 182)):
    """Paint a plausible rooftop rectangle with mild texture noise + a
    slight border to look like a real structure edge, not a flat sticker."""
    x0, y0 = cx - w // 2, cy - h // 2
    noise = rng.randint(-8, 8, size=(h, w, 3))
    patch = np.clip(np.array(base_color) + noise, 0, 255).astype(np.uint8)
    img[y0:y0+h, x0:x0+w] = patch
    cv2.rectangle(img, (x0, y0), (x0+w, y0+h), (140, 138, 132), 1)
    return (x0, y0, w, h)


# Two new synthetic structures placed in previously open/vegetated areas
new_1 = paint_synthetic_building(rgb_b, 150, 370, 22, 16, base_color=(200, 60, 55))   # reddish new roof in forest clearing
new_2 = paint_synthetic_building(rgb_b, 700, 460, 26, 20, base_color=(195, 193, 188))  # light RCC-style roof
print(f'Simulated new construction at {new_1} and {new_2}')

Image.fromarray(rgb_b).save(f'{OUT_DIR}/RGB_timeB_synthetic.png')

# =====================================================================
# 2. Run trained model on BOTH tiles (same inference function, reused)
# =====================================================================
torch.set_num_threads(4)
ckpt = torch.load(f'{OUT_DIR}/model_v2_ckpt.pt', map_location='cpu')
model = AttentionResUNet(in_ch=3, out_ch=1, base=16)
model.load_state_dict(ckpt['model'])
model.eval()


def run_inference(rgb_img_u8):
    rgb_f = rgb_img_u8.astype(np.float32) / 255.0
    patch, stride = 128, 96
    pred_sum = np.zeros((H, W), dtype=np.float32)
    pred_cnt = np.zeros((H, W), dtype=np.float32)
    ys = sorted(set(list(range(0, H - patch + 1, stride)) + [H - patch]))
    xs = sorted(set(list(range(0, W - patch + 1, stride)) + [W - patch]))
    with torch.no_grad():
        for yy in ys:
            for xx in xs:
                tile = rgb_f[yy:yy+patch, xx:xx+patch, :]
                t = torch.from_numpy(tile.transpose(2, 0, 1)).float().unsqueeze(0)
                out = torch.sigmoid(model(t)).squeeze().numpy()
                pred_sum[yy:yy+patch, xx:xx+patch] += out
                pred_cnt[yy:yy+patch, xx:xx+patch] += 1
    pred_cnt[pred_cnt == 0] = 1
    prob = pred_sum / pred_cnt
    mask = (prob > 0.5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    return clean


def vectorize(mask_u8, min_area=40):
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        area_px = cv2.contourArea(c)
        if area_px < min_area:
            continue
        eps = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True)
        coords = [[int(p[0]), int(p[1])] for p in approx.reshape(-1, 2)]
        if len(coords) < 3:
            continue
        polys.append({'polygon_px': coords, 'area_m2': float(round(area_px * GSD_M**2, 1))})
    return polys


print('Running inference on Time A...')
mask_a = run_inference(rgb_a)
polys_a = vectorize(mask_a)
print(f'Time A: {len(polys_a)} buildings detected')

print('Running inference on Time B...')
mask_b = run_inference(rgb_b)
polys_b = vectorize(mask_b)
print(f'Time B: {len(polys_b)} buildings detected')

# =====================================================================
# 3. Diff by IoU matching (rasterized, since polygons are irregular)
# =====================================================================
def poly_to_mask(poly, shape):
    m = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(m, [np.array(poly, dtype=np.int32)], 1)
    return m


def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0


masks_a = [poly_to_mask(p['polygon_px'], (H, W)) for p in polys_a]
masks_b = [poly_to_mask(p['polygon_px'], (H, W)) for p in polys_b]

new_construction = []
for i, mb in enumerate(masks_b):
    best_iou = max((iou(mb, ma) for ma in masks_a), default=0.0)
    if best_iou < 0.1:
        new_construction.append(polys_b[i])

demolished = []
for i, ma in enumerate(masks_a):
    best_iou = max((iou(ma, mb) for mb in masks_b), default=0.0)
    if best_iou < 0.1:
        demolished.append(polys_a[i])

print(f'\nChange detection result: {len(new_construction)} new construction candidates, '
      f'{len(demolished)} demolition candidates')

# =====================================================================
# 4. Visualization
# =====================================================================
overlay = rgb_b.copy()
for p in new_construction:
    pts = np.array(p['polygon_px'])
    cv2.drawContours(overlay, [pts], -1, (0, 255, 0), 3)
    cx, cy = pts.mean(axis=0).astype(int)
    cv2.putText(overlay, 'NEW', (cx-15, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

for p in demolished:
    pts = np.array(p['polygon_px'])
    cv2.drawContours(overlay, [pts], -1, (255, 0, 0), 3)
    cx, cy = pts.mean(axis=0).astype(int)
    cv2.putText(overlay, 'REMOVED', (cx-25, cy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

Image.fromarray(overlay).save(f'{OUT_DIR}/change_detection_overlay.png')

with open(f'{OUT_DIR}/change_detection.json', 'w') as f:
    json.dump({
        'time_a_buildings': len(polys_a), 'time_b_buildings': len(polys_b),
        'new_construction': new_construction, 'demolished': demolished,
    }, f, indent=2)

print('Saved change_detection_overlay.png and change_detection.json')
