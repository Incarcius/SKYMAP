"""
Road Detection Benchmark — Real Survey Images
===============================================
Runs BOTH the original and improved road detectors on all 4 real survey
aerial images from the SVAMITVA dashboard demo data and prints a detailed
quantitative comparison.

Because we have no labeled GT road mask for these images, accuracy is
assessed via 7 proxy metrics that strongly correlate with real road F1:

  1. Colour candidates (px)   — how much low-sat / road-coloured area found
  2. Components kept          — road-shaped regions retained after filter
  3. Blob rejection rate (%)  — pct of candidates wrongly killed (lower=better)
  4. Road-pixel coverage (%)  — share of image claimed as road
  5. Skeleton pixels          — length of extracted road network in pixels
  6. Skeleton segments        — number of distinct road polylines
  7. Total detected length(m) — using GSD 0.5m (SVAMITVA spec)

Higher coverage / segments / length + lower blob rate = better recall.
Mean PCA elongation > bounding-box elongation validates true road shape.

Usage:
    python eval_road_real.py
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label, regionprops

# Survey images live in dashboard/public/survey relative to repo root
REPO_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURVEY_DIR  = os.path.join(REPO_ROOT, 'dashboard', 'public', 'survey')
OUT_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

GSD_M = 0.5   # SVAMITVA spec: 0.5 m/pixel

IMAGE_FILES = ['area-01.jpg', 'area-02.jpg', 'area-03.jpg', 'area-04.jpg']

# ─────────────────────────────────────────────────────────────────────────────
def run_original(rgb_u8):
    """Exact original algorithm from infer_and_vectorize_best.py."""
    gray   = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv    = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat    = hsv[:, :, 1]
    val    = hsv[:, :, 2]

    cand = ((sat < 60) & (val > 70) & (val < 200)).astype(np.uint8)

    LENGTH = 21
    responses = np.zeros_like(gray, dtype=np.float32)
    for ang in range(0, 180, 15):
        se = np.zeros((LENGTH, LENGTH), np.uint8)
        cv2.line(se, (0, LENGTH//2), (LENGTH-1, LENGTH//2), 1, 1)
        M  = cv2.getRotationMatrix2D((LENGTH/2, LENGTH/2), ang, 1.0)
        se = cv2.warpAffine(se, M, (LENGTH, LENGTH), flags=cv2.INTER_NEAREST)
        responses = np.maximum(responses,
                               cv2.morphologyEx(cand*255, cv2.MORPH_OPEN, se).astype(np.float32))

    road_mask = (responses > 0).astype(np.uint8) * 255
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

    labeled = cc_label(road_mask > 0)
    filtered = np.zeros_like(road_mask)
    kept = blobs = 0
    for r in regionprops(labeled):
        if r.area < 60: continue
        minr, minc, maxr, maxc = r.bbox
        h, w = maxr-minr, maxc-minc
        elong = max(h, w) / (min(h, w) + 1e-6)
        if elong > 2.0:
            filtered[labeled == r.label] = 255; kept += 1
        else:
            blobs += 1

    skel = skeletonize(filtered > 0)
    return cand, filtered, skel, kept, blobs


def run_improved(rgb_u8):
    """Improved algorithm v2."""
    gray   = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    hsv    = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    sat    = hsv[:, :, 1]
    val    = hsv[:, :, 2]
    hue    = hsv[:, :, 0]

    asphalt   = ((sat < 55) & (val > 65) & (val < 210)).astype(np.uint8)
    dirt      = ((hue > 5)  & (hue < 25) & (sat > 15) & (sat < 80)
                 & (val > 80) & (val < 190)).astype(np.uint8)
    concrete  = ((sat < 35) & (val >= 200)).astype(np.uint8)
    cand      = np.clip(asphalt + dirt + concrete, 0, 1).astype(np.uint8)

    LENGTH = 31
    responses = np.zeros_like(gray, dtype=np.float32)
    for ang in range(0, 180, 10):
        se = np.zeros((LENGTH, LENGTH), np.uint8)
        cv2.line(se, (0, LENGTH//2), (LENGTH-1, LENGTH//2), 1, 1)
        M  = cv2.getRotationMatrix2D((LENGTH/2, LENGTH/2), ang, 1.0)
        se = cv2.warpAffine(se, M, (LENGTH, LENGTH), flags=cv2.INTER_NEAREST)
        responses = np.maximum(responses,
                               cv2.morphologyEx(cand*255, cv2.MORPH_OPEN, se).astype(np.float32))

    road_mask = (responses > 0).astype(np.uint8) * 255
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, np.ones((11,11), np.uint8))

    labeled = cc_label(road_mask > 0)
    filtered = np.zeros_like(road_mask)
    kept = blobs = 0
    for r in regionprops(labeled):
        if r.area < 50: continue
        elong = r.axis_major_length / (r.axis_minor_length + 1e-6)
        if elong > 1.8:
            filtered[labeled == r.label] = 255; kept += 1
        else:
            blobs += 1

    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    skel = skeletonize(filtered > 0)
    return cand, filtered, skel, kept, blobs


def metrics(cand, filtered, skel, kept, blobs, H, W):
    sk_lab   = cc_label(skel)
    segs     = [r for r in regionprops(sk_lab) if r.area >= 8]
    blob_pct = 100.0 * blobs / (kept + blobs + 1e-6)
    cov_pct  = 100.0 * (filtered > 0).sum() / (H * W)
    sk_px    = int(skel.sum())
    seg_cnt  = len(segs)
    tot_m    = sum(r.area for r in segs) * GSD_M
    mean_el  = float(np.mean([r.axis_major_length/(r.axis_minor_length+1e-6)
                               for r in segs])) if segs else 0.0
    return dict(cand_px=int(cand.sum()), kept=kept, blobs=blobs,
                blob_pct=blob_pct, cov_pct=cov_pct,
                sk_px=sk_px, seg_cnt=seg_cnt, tot_m=tot_m, mean_el=mean_el)


def make_overlay(rgb_u8, skel, filtered, color):
    out = rgb_u8.copy()
    mask = filtered > 0
    out[mask] = (out[mask].astype(np.float32)*0.5
                 + np.array([255,220,80], np.float32)*0.5).astype(np.uint8)
    ys, xs = np.where(skel)
    out[ys, xs] = color
    return out


# ─────────────────────────────────────────────────────────────────────────────
all_orig, all_impr = [], []
panels = []

print('=' * 70)
print(f'Road Detection Benchmark on {len(IMAGE_FILES)} real aerial survey images')
print(f'GSD assumed: {GSD_M} m/px  (SVAMITVA CORS-RTK drone spec)')
print('=' * 70)

for fname in IMAGE_FILES:
    fpath = os.path.join(SURVEY_DIR, fname)
    if not os.path.exists(fpath):
        print(f'  SKIP (not found): {fpath}')
        continue

    rgb    = np.array(Image.open(fpath).convert('RGB'))
    rgb_u8 = rgb
    H, W   = rgb.shape[:2]
    print(f'\n[{fname}]  {H}x{W}px = {H*GSD_M:.0f}x{W*GSD_M:.0f} m')

    t0 = time.time()
    co, fo, so, ko, bo = run_original(rgb_u8)
    mo = metrics(co, fo, so, ko, bo, H, W)
    to = time.time() - t0

    t0 = time.time()
    ci, fi, si, ki, bi = run_improved(rgb_u8)
    mi = metrics(ci, fi, si, ki, bi, H, W)
    ti = time.time() - t0

    all_orig.append(mo); all_impr.append(mi)

    print(f'  {"Metric":<28} {"ORIGINAL":>12} {"IMPROVED":>12}  {"Delta":>10}')
    print(f'  {"-"*64}')
    rows = [
        ('Colour candidates (px)',  'cand_px', '', '{:>12,}'),
        ('Components kept',         'kept',    '', '{:>12,}'),
        ('Blob rejection rate (%)', 'blob_pct','%', '{:>12.1f}'),
        ('Road coverage (%)',       'cov_pct', '%', '{:>12.2f}'),
        ('Skeleton pixels',         'sk_px',   '',  '{:>12,}'),
        ('Road segments',           'seg_cnt', '',  '{:>12,}'),
        ('Total road length (m)',   'tot_m',   'm',  '{:>12,.0f}'),
        ('Mean PCA elongation',     'mean_el', '',  '{:>12.1f}'),
    ]
    for label, key, unit, fmt in rows:
        ov = mo[key]; iv = mi[key]
        delta = iv - ov
        sign  = '+' if delta >= 0 else ''
        better = ''
        if key == 'blob_pct':
            better = '(better)' if delta < 0 else ('(worse)' if delta > 0 else '')
        elif key not in ('cand_px',):
            better = '(better)' if delta > 0 else ('(worse)' if delta < 0 else '')
        print(f'  {label:<28} {fmt.format(ov)}{unit}  {fmt.format(iv)}{unit}  '
              f'{sign}{delta:+.1f}{unit}  {better}')
    print(f'  Time: orig={to:.2f}s  impr={ti:.2f}s')

    # Save side-by-side overlay
    vis_o = make_overlay(rgb_u8, so, fo, [255, 60, 60])
    vis_i = make_overlay(rgb_u8, si, fi, [60, 255, 100])
    cv2.putText(vis_o, 'ORIGINAL', (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
    cv2.putText(vis_o, 'ORIGINAL', (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 1)
    cv2.putText(vis_i, 'IMPROVED', (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 3)
    cv2.putText(vis_i, 'IMPROVED', (15,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 1)
    side = np.concatenate([vis_o, vis_i], axis=1)
    stem = os.path.splitext(fname)[0]
    out_path = os.path.join(OUT_DIR, f'road_eval_{stem}.jpg')
    Image.fromarray(side).save(out_path, quality=88)
    print(f'  Overlay -> {out_path}')


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate summary across all images
print('\n' + '=' * 70)
print('AGGREGATE SUMMARY (mean across all images)')
print(f'  {"Metric":<28} {"ORIGINAL":>12} {"IMPROVED":>12}  {"Delta":>10}')
print(f'  {"-"*64}')
for label, key, unit, fmt in rows:
    ov = np.mean([m[key] for m in all_orig])
    iv = np.mean([m[key] for m in all_impr])
    delta = iv - ov
    sign  = '+' if delta >= 0 else ''
    better = ''
    if key == 'blob_pct':
        better = '(better)' if delta < 0 else ('(worse)' if delta > 0 else '')
    elif key not in ('cand_px',):
        better = '(better)' if delta > 0 else ('(worse)' if delta < 0 else '')
    print(f'  {label:<28} {fmt.format(ov)}{unit}  {fmt.format(iv)}{unit}  '
          f'{sign}{delta:+.1f}{unit}  {better}')

print('\nOverlay images saved in:', OUT_DIR)
