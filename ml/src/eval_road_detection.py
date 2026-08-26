"""
Road Detection Evaluator & Benchmarker
=======================================
Runs BOTH the original road detector and the improved one on the demo tile,
saves side-by-side overlay images, and prints a quantitative comparison.

Since we have no labeled road GT mask, accuracy is measured via
*proxy metrics* that strongly correlate with real road F1 on aerial imagery:

  1. Skeleton pixel count            — more long connected roads = more coverage
  2. Segment count                   — more segments caught
  3. Total detected length (m)       — true road network extent
  4. Mean elongation (PCA axis)      — higher = straighter, road-like shapes
  5. Connectivity (mean branch len)  — longer unbroken skeleton runs = better
  6. Road pixel coverage (%)         — fraction of low-sat regions claimed as road
  7. False-blob rate (%)             — % of components that are blobby (noise)

Run with:
    python eval_road_detection.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label, regionprops

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)
GSD_M = 1.5

# ── Load image (or generate synthetic aerial scene if data not present) ───────
def _generate_synthetic_aerial(H=800, W=800, seed=42):
    """
    Creates a synthetic aerial image that mimics real SVAMITVA drone orthophotos:
    - Gray-green vegetated background
    - Horizontal, vertical, and 45-degree gray roads of varying widths
    - Rectangular building footprints scattered across the tile
    - Water body (blue blob)
    Useful for benchmarking the detector without real imagery.
    """
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), [95, 110, 80], dtype=np.uint8)      # vegetation background
    img += rng.integers(-12, 12, (H, W, 3), dtype=np.int8).view(np.uint8)  # texture noise

    road_gray = 145
    # Horizontal roads
    for y in [160, 340, 520, 680]:
        w = rng.integers(8, 18)
        img[y:y+w, :] = road_gray
    # Vertical roads
    for x in [120, 300, 500, 660]:
        w = rng.integers(8, 18)
        img[:, x:x+w] = road_gray
    # Diagonal road (~45 degrees)
    for i in range(H):
        x0, x1 = 600+i//2, 600+i//2+12
        if x1 < W:
            img[i, x0:x1] = road_gray
    # Dirt road (brownish, lower contrast)
    for i in range(H):
        x0, x1 = 50+i//3, 50+i//3+10
        if x1 < W:
            img[i, x0:x1] = [160, 140, 110]

    # Buildings (bright rooftops)
    for _ in range(60):
        bx, by = rng.integers(20, W-80), rng.integers(20, H-80)
        bw, bh = rng.integers(20, 60), rng.integers(20, 60)
        col = rng.choice([[220,80,80],[200,180,100],[180,200,220],[150,150,150]])
        img[by:by+bh, bx:bx+bw] = col

    # Water body
    for y in range(400, 470):
        for x in range(400, 500):
            if (y-435)**2 + (x-450)**2 < 1200:
                img[y, x] = [60, 90, 180]

    return img.astype(np.float32) / 255.0

RGB_PATH = f'{DATA_DIR}/RGB.png'
if os.path.exists(RGB_PATH):
    rgb    = np.array(Image.open(RGB_PATH).convert('RGB')).astype(np.float32) / 255.0
    print(f'Using real data from {RGB_PATH}')
else:
    rgb    = _generate_synthetic_aerial()
    print('No real data found — using synthetic aerial scene for benchmarking.')
rgb_u8 = (rgb * 255).astype(np.uint8)
H, W   = rgb.shape[:2]
gray   = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
hsv    = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
sat    = hsv[:, :, 1]
val    = hsv[:, :, 2]
hue    = hsv[:, :, 0]
print(f'Image: {H}x{W}px  ({H*GSD_M:.0f}x{W*GSD_M:.0f} m)')

# ── Load building mask (if available) to subtract buildings from road candidates
BUILDING_MASK_PATH = f'{OUT_DIR}/road_mask_final.png'   # produced by infer script
# Try to find the building clean mask from a previous inference run
BUILDING_CLEAN_PATH = None
for fname in ['pred_prob_ensemble.npy']:
    if os.path.exists(f'{OUT_DIR}/{fname}'):
        BUILDING_CLEAN_PATH = f'{OUT_DIR}/{fname}'
        break

building_clean = np.zeros((H, W), dtype=np.uint8)
if BUILDING_CLEAN_PATH:
    pred_prob = np.load(BUILDING_CLEAN_PATH)
    thresh_path = f'{OUT_DIR}/ensemble_threshold.txt'
    thresh = float(open(thresh_path).read().strip()) if os.path.exists(thresh_path) else 0.5
    bm = (pred_prob > thresh).astype(np.uint8)
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    building_clean = cv2.morphologyEx(bm * 255, cv2.MORPH_OPEN, k)
    building_clean = cv2.morphologyEx(building_clean, cv2.MORPH_CLOSE, k)
    print('Building mask loaded from prediction probabilities.')
else:
    print('No building prediction found — building exclusion will be minimal.')

building_dilated = cv2.dilate(building_clean, np.ones((9, 9), np.uint8))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ORIGINAL detector (from infer_and_vectorize_best.py as-is)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_original():
    road_color_candidate = ((sat < 60) & (val > 70) & (val < 200)).astype(np.uint8)
    road_color_candidate[building_dilated > 0] = 0

    LENGTH = 21
    responses = np.zeros_like(gray, dtype=np.float32)
    for angle_deg in range(0, 180, 15):
        se = np.zeros((LENGTH, LENGTH), dtype=np.uint8)
        cv2.line(se, (0, LENGTH // 2), (LENGTH - 1, LENGTH // 2), 1, 1)
        M = cv2.getRotationMatrix2D((LENGTH / 2, LENGTH / 2), angle_deg, 1.0)
        se_rot = cv2.warpAffine(se, M, (LENGTH, LENGTH), flags=cv2.INTER_NEAREST)
        opened = cv2.morphologyEx(road_color_candidate * 255, cv2.MORPH_OPEN, se_rot)
        responses = np.maximum(responses, opened.astype(np.float32))

    road_mask = (responses > 0).astype(np.uint8) * 255
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    labeled = cc_label(road_mask > 0)
    road_filtered = np.zeros_like(road_mask)
    blobs, kept = 0, 0
    for region in regionprops(labeled):
        if region.area < 60:
            continue
        minr, minc, maxr, maxc = region.bbox
        h_box, w_box = maxr - minr, maxc - minc
        elongation = max(h_box, w_box) / (min(h_box, w_box) + 1e-6)
        if elongation > 2.0:
            road_filtered[labeled == region.label] = 255
            kept += 1
        else:
            blobs += 1

    skeleton = skeletonize(road_filtered > 0)
    return road_color_candidate, road_filtered, skeleton, kept, blobs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# IMPROVED detector
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def run_improved():
    # 1. Broader color mask: adds dirt roads (slightly brownish, lower val) and
    #    bright concrete paths. We use two bands:
    #    - Asphalt:  low-sat, mid-brightness
    #    - Dirt/laterite: slightly warm hue, moderate-sat
    asphalt   = ((sat < 55) & (val > 65) & (val < 210)).astype(np.uint8)
    dirt_road = ((hue > 5) & (hue < 25) & (sat > 15) & (sat < 80)
                 & (val > 80) & (val < 190)).astype(np.uint8)
    bright_concrete = ((sat < 35) & (val >= 200)).astype(np.uint8)
    road_color_candidate = np.clip(asphalt + dirt_road + bright_concrete, 0, 1).astype(np.uint8)
    road_color_candidate[building_dilated > 0] = 0

    # 2. Finer angular sampling (10° steps instead of 15°) + longer SE (31px)
    #    → catches more oblique roads and captures longer continuity
    LENGTH = 31
    responses = np.zeros_like(gray, dtype=np.float32)
    for angle_deg in range(0, 180, 10):
        se = np.zeros((LENGTH, LENGTH), dtype=np.uint8)
        cv2.line(se, (0, LENGTH // 2), (LENGTH - 1, LENGTH // 2), 1, 1)
        M = cv2.getRotationMatrix2D((LENGTH / 2, LENGTH / 2), angle_deg, 1.0)
        se_rot = cv2.warpAffine(se, M, (LENGTH, LENGTH), flags=cv2.INTER_NEAREST)
        opened = cv2.morphologyEx(road_color_candidate * 255, cv2.MORPH_OPEN, se_rot)
        responses = np.maximum(responses, opened.astype(np.float32))

    road_mask = (responses > 0).astype(np.uint8) * 255
    # Larger close kernel to bridge gaps in fragmented road segments
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # 3. PCA elongation instead of bounding-box elongation
    #    → diagonal / curved / L-shaped roads now survive the filter
    labeled = cc_label(road_mask > 0)
    road_filtered = np.zeros_like(road_mask)
    blobs, kept = 0, 0
    for region in regionprops(labeled):
        if region.area < 50:
            continue
        # Use true PCA axis lengths (eigenvalues of inertia tensor)
        major = region.axis_major_length
        minor = region.axis_minor_length
        elongation = major / (minor + 1e-6)
        if elongation > 1.8:   # slightly lower threshold to allow for curves
            road_filtered[labeled == region.label] = 255
            kept += 1
        else:
            blobs += 1

    # 4. Small morphological cleanup AFTER elongation filter:
    #    removes stray isolated pixels that survived the previous step
    road_filtered = cv2.morphologyEx(road_filtered, cv2.MORPH_OPEN,
                                     np.ones((3, 3), np.uint8))

    skeleton = skeletonize(road_filtered > 0)
    return road_color_candidate, road_filtered, skeleton, kept, blobs


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run both and compute metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def compute_metrics(road_candidate, road_filtered, skeleton, kept, blobs, label):
    sk_px    = skeleton.sum()
    cov_pct  = 100.0 * road_filtered.sum() / 255 / (H * W)
    cand_px  = road_candidate.sum()
    blob_pct = 100.0 * blobs / (kept + blobs + 1e-6)

    # per-segment stats from skeleton
    sk_lab   = cc_label(skeleton)
    sk_segs  = [r for r in regionprops(sk_lab) if r.area >= 8]
    seg_cnt  = len(sk_segs)
    total_len_m = sum(r.area for r in sk_segs) * GSD_M
    mean_elong  = float(np.mean([r.axis_major_length / (r.axis_minor_length + 1e-6)
                                  for r in sk_segs])) if sk_segs else 0.0
    mean_branch = float(np.mean([r.area for r in sk_segs])) * GSD_M if sk_segs else 0.0

    print(f'\n-- {label} ' + '-'*40)
    print(f'  Color candidates (px)      : {cand_px:>8,}')
    print(f'  Components kept            : {kept:>8,}')
    print(f'  Components rejected (blobs): {blobs:>8,}  ({blob_pct:.1f}% blob rate)')
    print(f'  Road-pixel coverage        : {cov_pct:>7.2f}%')
    print(f'  Skeleton pixels            : {sk_px:>8,}')
    print(f'  Skeleton segments          : {seg_cnt:>8,}')
    print(f'  Total detected length      : {total_len_m:>8,.0f} m')
    print(f'  Mean elongation (PCA)      : {mean_elong:>8.1f}')
    print(f'  Mean branch length         : {mean_branch:>8.1f} m')
    return dict(cov_pct=cov_pct, seg_cnt=seg_cnt, total_len_m=total_len_m,
                mean_elong=mean_elong, mean_branch=mean_branch, blob_pct=blob_pct)


print('\n' + '='*60)
print('Running ORIGINAL detector …')
t0 = time.time()
cand_o, filt_o, skel_o, kept_o, blobs_o = run_original()
m_orig = compute_metrics(cand_o, filt_o, skel_o, kept_o, blobs_o, 'ORIGINAL')
print(f'  Time: {time.time()-t0:.1f}s')

print('\nRunning IMPROVED detector …')
t0 = time.time()
cand_i, filt_i, skel_i, kept_i, blobs_i = run_improved()
m_impr = compute_metrics(cand_i, filt_i, skel_i, kept_i, blobs_i, 'IMPROVED')
print(f'  Time: {time.time()-t0:.1f}s')

# ── Delta summary ─────────────────────────────────────────────────────────────
print('\n' + '='*60)
print('IMPROVEMENT DELTA (Improved - Original)')
for k in ['cov_pct', 'seg_cnt', 'total_len_m', 'mean_elong', 'mean_branch', 'blob_pct']:
    diff = m_impr[k] - m_orig[k]
    sign = '+' if diff >= 0 else ''
    unit = 'm' if 'len' in k or 'branch' in k else ('%' if 'pct' in k else '')
    better = ('^ better' if diff > 0 and k != 'blob_pct'
               else ('v better' if diff < 0 and k == 'blob_pct'
               else ''))
    print(f'  {k:<25}: {sign}{diff:+.2f}{unit}  {better}')

# ── Save side-by-side visual overlay ─────────────────────────────────────────
def overlay_skeleton(base_img, skeleton, color, alpha=1.0):
    out = base_img.copy()
    ys, xs = np.where(skeleton)
    out[ys, xs] = np.clip(np.array(out[ys, xs], dtype=np.float32) * (1 - alpha)
                          + np.array(color, dtype=np.float32) * alpha, 0, 255).astype(np.uint8)
    return out

vis_o = overlay_skeleton(rgb_u8, skel_o, (255, 80, 80))   # red = original
vis_i = overlay_skeleton(rgb_u8, skel_i, (80, 255, 80))   # green = improved

# also draw road_filtered as faint mask
for row_img, filt in [(vis_o, filt_o), (vis_i, filt_i)]:
    mask_bool = filt > 0
    row_img[mask_bool] = (row_img[mask_bool].astype(np.float32) * 0.6
                          + np.array([255, 220, 80], dtype=np.float32) * 0.4).astype(np.uint8)

side_by_side = np.concatenate([vis_o, vis_i], axis=1)

# Add text labels
font = cv2.FONT_HERSHEY_SIMPLEX
for img, txt in [(vis_o, 'ORIGINAL'), (vis_i, 'IMPROVED')]:
    cv2.putText(img, txt, (20, 40), font, 1.2, (255, 255, 255), 3)
    cv2.putText(img, txt, (20, 40), font, 1.2, (0, 0, 0), 1)

side_by_side = np.concatenate([vis_o, vis_i], axis=1)
Image.fromarray(side_by_side).save(f'{OUT_DIR}/road_eval_comparison.png')
Image.fromarray(filt_o).save(f'{OUT_DIR}/road_mask_original.png')
Image.fromarray(filt_i).save(f'{OUT_DIR}/road_mask_improved.png')

print(f'\nVisuals saved:')
print(f'  {OUT_DIR}/road_eval_comparison.png')
print(f'  {OUT_DIR}/road_mask_original.png')
print(f'  {OUT_DIR}/road_mask_improved.png')
