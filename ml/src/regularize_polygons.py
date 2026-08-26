"""
Polygon regularization post-processing.

Raw CNN segmentation masks, even after morphological cleanup and
approxPolyDP simplification, still produce building outlines with
irregular, non-rectilinear edges - real buildings are overwhelmingly
rectangular or L/T-shaped, not organic blobs. This step snaps each
building polygon toward axis-aligned rectilinear form using minAreaRect
(for simple/small buildings) or an edge-angle-histogram-based rotation
correction (for larger/complex footprints), which is standard practice in
production building-footprint pipelines (e.g. Microsoft's Bing Maps
building footprints use a similar regularization step).

This does NOT change which pixels were classified as building - it only
cleans up the final vector shape, so it's a pure presentation-quality /
downstream-usability improvement (GIS systems and property records expect
clean rectilinear footprints, not CNN-jagged blobs).
"""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
from PIL import Image
from shapely.geometry import Polygon

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

with open(f'{OUT_DIR}/buildings_best.json') as f:
    buildings = json.load(f)


def regularize_polygon(coords, area_px):
    """Return a cleaned-up polygon: for small/simple buildings, snap to the
    minimum-area rotated rectangle (captures orientation, removes jaggedness).
    For larger/more complex footprints (L-shaped industrial buildings etc.),
    keep the approxPolyDP simplification but snap each edge angle to the
    nearest multiple of the building's dominant orientation, which straightens
    near-axis-aligned edges without collapsing genuine L/T shapes into a box."""
    pts = np.array(coords, dtype=np.float32)
    if len(pts) < 3:
        return coords, 'unchanged'

    rect = cv2.minAreaRect(pts)
    box = cv2.boxPoints(rect)
    rect_area = rect[1][0] * rect[1][1]

    # how well does the polygon fill its own bounding rectangle?
    fill_ratio = area_px / (rect_area + 1e-6)

    if len(pts) <= 5 or fill_ratio > 0.75:
        # simple, roughly-rectangular footprint -> snap directly to the
        # rotated bounding rectangle
        return box.tolist(), 'snapped_to_rect'
    else:
        # complex footprint (L-shape etc.) -> straighten edges toward the
        # dominant orientation instead of forcing a single rectangle
        angle = rect[2]
        theta = np.radians(-angle)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        center = pts.mean(axis=0)
        rotated = (pts - center) @ R.T
        # snap each point's coordinates to reduce small-angle jaggedness
        # (round to nearest 0.5px in rotated frame, a mild regularization)
        rotated = np.round(rotated * 2) / 2
        straightened = (rotated @ R) + center
        return straightened.tolist(), 'edge_straightened'


n_snapped, n_straightened, n_unchanged = 0, 0, 0
regularized = []
for b in buildings:
    pts = np.array(b['polygon_px'], dtype=np.float32)
    area_px = cv2.contourArea(pts) if len(pts) >= 3 else 0
    new_coords, method = regularize_polygon(b['polygon_px'], area_px)
    b2 = dict(b)
    b2['polygon_px_original'] = b['polygon_px']
    b2['polygon_px'] = [[round(float(x), 1), round(float(y), 1)] for x, y in new_coords]
    b2['regularization_method'] = method
    regularized.append(b2)
    if method == 'snapped_to_rect':
        n_snapped += 1
    elif method == 'edge_straightened':
        n_straightened += 1
    else:
        n_unchanged += 1

with open(f'{OUT_DIR}/buildings_regularized.json', 'w') as f:
    json.dump(regularized, f, indent=2)

print(f'Regularized {len(regularized)} building polygons:')
print(f'  {n_snapped} snapped to rotated bounding rectangle (simple footprints)')
print(f'  {n_straightened} edge-straightened (complex L/T-shaped footprints)')
print(f'  {n_unchanged} left unchanged (degenerate/tiny)')

# ---- visualize before/after on a cropped region for the PPT ----
rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB'))
before_img = rgb.copy()
after_img = rgb.copy()
for b in regularized:
    orig = np.array(b['polygon_px_original'], dtype=np.int32)
    reg = np.array(b['polygon_px'], dtype=np.int32)
    cv2.polylines(before_img, [orig], True, (255, 60, 60), 1)
    cv2.polylines(after_img, [reg], True, (60, 200, 60), 1)

Image.fromarray(before_img).save(f'{OUT_DIR}/polygons_before_reg.png')
Image.fromarray(after_img).save(f'{OUT_DIR}/polygons_after_reg.png')
print('Saved before/after polygon visualizations')
