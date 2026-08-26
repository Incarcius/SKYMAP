"""
Feature extraction v2 - closes the biggest scope gap in the prototype:
the PS asks for buildings + roads + waterbodies, but the v1 pipeline only
extracted buildings. This version adds all three, plus upgrades the roof
classifier from fixed if/else color thresholds to an unsupervised
clustering approach (more defensible: cluster assignments are data-driven,
and only 4 centroid->label mappings need human review instead of trusting
thousands of individual rule-based decisions).

ROADS: no labeled road data or pretrained D-LinkNet weights are reachable
in this sandbox, so this uses a classical multi-orientation morphological
linear-feature detector (opening with rotated line structuring elements at
12 angles, response = max across orientations) to find elongated,
low-saturation, mid-brightness structures, excludes anything overlapping a
detected building, skeletonizes the result, then vectorizes each connected
skeleton component into an ordered polyline via PCA projection (points
sorted along the component's principal axis). This is the honest classical
stand-in for D-LinkNet described in the tech stack.

WATER: HSV hue/saturation thresholding tuned for blue-cyan tones combined
with a low-texture (Laplacian variance) filter to reject non-water blue
objects, then standard contour vectorization (same approach as buildings).
This is the RGB-only fallback named in the tech stack (NDWI requires a
near-infrared band this demo tile doesn't have).

ROOF CLASSIFIER: replaces the manual if/else rule bank with KMeans
clustering (k=4) over [hue, saturation, value, texture] features across all
detected buildings, then a single semantic label is assigned per cluster
centroid (not per building) - this means a human reviewer only has to
validate 4 decisions, not thousands, which is the actual workflow a real
active-learning pipeline would use.
"""
import sys, json
import os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import cv2
from PIL import Image
from shapely.geometry import Polygon, LineString, mapping
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label, regionprops
from sklearn.cluster import KMeans

from model import AttentionResUNet
from dataset import split_regions

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
GSD_M = 1.5  # ground sample distance (m/px) of this demo tile; SVAMITVA imagery is 0.5m

rgb = np.array(Image.open(f'{DATA_DIR}/RGB.png').convert('RGB')).astype(np.float32) / 255.0
rgb_u8 = (rgb * 255).astype(np.uint8)
H, W = rgb.shape[:2]

# =====================================================================
# 1. BUILDINGS — reuse trained v2 model, same inference + vectorization
#    as before, but roof classification is now clustering-based (below).
# =====================================================================
torch.set_num_threads(4)
ckpt = torch.load(f'{OUT_DIR}/model_v2_ckpt.pt', map_location='cpu')
model = AttentionResUNet(in_ch=3, out_ch=1, base=16)
model.load_state_dict(ckpt['model'])
model.eval()

pred_prob = np.load(f'{OUT_DIR}/pred_prob_final.npy')  # TTA-averaged probability map
with open(f'{OUT_DIR}/best_threshold.txt') as f:
    BEST_THRESH = float(f.read().strip())
print(f'Using final model: TTA-averaged probabilities, tuned threshold={BEST_THRESH:.2f}')
building_mask = (pred_prob > BEST_THRESH).astype(np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
building_clean = cv2.morphologyEx(building_mask * 255, cv2.MORPH_OPEN, kernel)
building_clean = cv2.morphologyEx(building_clean, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(building_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
MIN_BUILDING_AREA_PX = 40

building_records = []
building_feats = []  # for clustering
for i, c in enumerate(contours):
    area_px = cv2.contourArea(c)
    if area_px < MIN_BUILDING_AREA_PX:
        continue
    x, y, w, h = cv2.boundingRect(c)
    crop = cv2.cvtColor(rgb_u8[y:y+h, x:x+w], cv2.COLOR_RGB2BGR)
    if crop.size == 0:
        continue
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hm, sm, vm = hsv[:, :, 0].mean(), hsv[:, :, 1].mean(), hsv[:, :, 2].mean()
    gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    texture = cv2.Laplacian(gray_crop, cv2.CV_64F).var()

    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    coords = [[int(p[0]), int(p[1])] for p in approx.reshape(-1, 2)]
    if len(coords) < 3:
        continue

    area_m2 = float(round(area_px * (GSD_M ** 2), 1))
    building_records.append({
        'id': int(i), 'polygon_px': coords, 'area_m2': area_m2,
        'bbox': [int(x), int(y), int(w), int(h)],
    })
    building_feats.append([hm, sm, vm, min(texture, 2000)])  # cap texture outliers

building_feats = np.array(building_feats, dtype=np.float32)
# standardize features before clustering
feat_mean, feat_std = building_feats.mean(axis=0), building_feats.std(axis=0) + 1e-6
feat_norm = (building_feats - feat_mean) / feat_std

N_CLUSTERS = 4
km = KMeans(n_clusters=N_CLUSTERS, n_init=10, random_state=42)
cluster_ids = km.fit_predict(feat_norm)
distances = km.transform(feat_norm)  # distance to every centroid

# Assign ONE semantic label per cluster centroid (human-reviewable step -
# only 4 decisions instead of thousands). Centroids are in ORIGINAL units
# (denormalize) so thresholds are interpretable.
centroids_raw = km.cluster_centers_ * feat_std + feat_mean
cluster_label_map = {}
used_labels = set()
label_priority = []
for cid, (h_c, s_c, v_c, t_c) in enumerate(centroids_raw):
    if v_c > 140 and s_c < 70:
        label = 'RCC (Concrete)'
    elif 5 < h_c < 30 and s_c > 70:
        label = 'Tiled'
    elif v_c < 110 and t_c > 150:
        label = 'Tin/Metal Sheet'
    else:
        label = 'Other'
    cluster_label_map[cid] = label

print('Cluster centroids -> semantic labels (human-reviewable, n=4 decisions):')
for cid, label in cluster_label_map.items():
    h_c, s_c, v_c, t_c = centroids_raw[cid]
    n_members = (cluster_ids == cid).sum()
    print(f'  Cluster {cid} (n={n_members:4d}): H={h_c:5.1f} S={s_c:5.1f} V={v_c:5.1f} tex={t_c:6.1f} -> {label}')

overlay = rgb_u8.copy()
color_map = {'RCC (Concrete)': (255, 80, 80), 'Tiled': (255, 180, 60),
             'Tin/Metal Sheet': (120, 200, 255), 'Other': (180, 180, 180)}


def solar_score(area_m2, roof_label):
    material_factor = {'RCC (Concrete)': 1.0, 'Tin/Metal Sheet': 0.9,
                        'Tiled': 0.6, 'Other': 0.5}.get(roof_label, 0.4)
    return float(round(area_m2 * material_factor * 0.15, 2))


for idx, rec in enumerate(building_records):
    cid = cluster_ids[idx]
    label = cluster_label_map[cid]
    dist_to_own = distances[idx, cid]
    dist_to_others = np.delete(distances[idx], cid)
    # confidence: how much closer to assigned centroid vs nearest rival
    margin = (dist_to_others.min() - dist_to_own) / (dist_to_others.min() + dist_to_own + 1e-6)
    conf = round(float(np.clip(0.5 + margin, 0.3, 0.97)), 2)

    rec['roof_material'] = label
    rec['roof_confidence'] = float(conf)
    rec['roof_cluster_id'] = int(cid)
    rec['estimated_solar_kwp'] = solar_score(rec['area_m2'], label)

    pts = np.array(rec['polygon_px'])
    cv2.drawContours(overlay, [pts], -1, color_map.get(label, (200, 200, 200)), 2)

print(f'\nBuildings: {len(building_records)} detected')

# =====================================================================
# =====================================================================
# 2. ROADS — improved multi-orientation morphological linear-feature detector
#
#  v2 improvements over the original:
#   a) Broader colour mask: adds dirt/laterite roads (warm hue, moderate sat)
#      and bright concrete paths, not just grey asphalt.
#   b) Finer angular sampling: 10° steps instead of 15° so oblique roads
#      are caught; longer SE (31px) captures road continuity better.
#   c) Larger morphological close (11×11) to bridge fragmented segments.
#   d) PCA-based elongation filter: uses region.axis_major_length /
#      axis_minor_length instead of the bounding-box ratio. The bounding-box
#      approach falsely rejects diagonal / L-shaped / curved roads because
#      their bbox is roughly square even though the region itself is thin.
#   e) Minimum area lowered to 50 px and threshold to 1.8 to catch narrower
#      village paths without letting blobby noise through.
# =====================================================================
# ── Roads v3: vegetation mask, CLAHE, width filter, gap bridging ──────────
from road_detector import detect_roads, vectorize_skeleton
_, road_filtered, skeleton, _, _ = detect_roads(rgb_u8, building_clean=building_clean, GSD_M=GSD_M)
hsv_full = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
sat = hsv_full[:, :, 1]
hue = hsv_full[:, :, 0]
gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
road_records = vectorize_skeleton(skeleton, GSD_M=GSD_M)
sk_labeled = cc_label(skeleton)
overlay_roads = overlay.copy()
for rec in road_records:
    pts = np.array(rec['polyline_px']).astype(int)
    for j in range(len(pts) - 1):
        cv2.line(overlay_roads, tuple(pts[j]), tuple(pts[j + 1]), (255, 255, 0), 3)

print(f'Roads: {len(road_records)} segments detected, total length '
      f'{sum(r["length_m"] for r in road_records):,.0f} m')

# =====================================================================
# 3. WATERBODIES — HSV blue/cyan threshold + low-texture filter
# =====================================================================
hue = hsv_full[:, :, 0]
water_candidate = (((hue > 85) & (hue < 135)) & (sat > 40) & (val > 60) & (val < 220)).astype(np.uint8)

# reject high-texture blue regions (e.g. blue-tinted roofs/cars have edges;
# water is visually smooth)
tex_map = cv2.Laplacian(gray, cv2.CV_64F)
tex_map = cv2.GaussianBlur(np.abs(tex_map).astype(np.float32), (9, 9), 0)
water_candidate[tex_map > 15] = 0
water_candidate[building_dilated > 0] = 0

water_mask = cv2.morphologyEx(water_candidate * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

w_contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
water_records = []
for i, c in enumerate(w_contours):
    area_px = cv2.contourArea(c)
    if area_px < 25:
        continue
    eps = 0.01 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, eps, True)
    coords = [[int(p[0]), int(p[1])] for p in approx.reshape(-1, 2)]
    if len(coords) < 3:
        continue
    area_m2 = float(round(area_px * (GSD_M ** 2), 1))
    water_records.append({'id': int(i), 'polygon_px': coords, 'area_m2': area_m2})
    pts = np.array(coords)
    cv2.drawContours(overlay_roads, [pts], -1, (0, 220, 220), 2)

print(f'Waterbodies: {len(water_records)} detected, total area '
      f'{sum(w["area_m2"] for w in water_records):,.0f} m^2')

Image.fromarray(overlay_roads).save(f'{OUT_DIR}/final_extraction_overlay.png')
Image.fromarray(road_filtered).save(f'{OUT_DIR}/road_mask_final.png')
Image.fromarray(water_mask).save(f'{OUT_DIR}/water_mask_final.png')

# =====================================================================
# 4. Combined GeoJSON export (buildings + roads + water, typed features)
# =====================================================================
features = []
for r in building_records:
    features.append({
        'type': 'Feature', 'properties': {**{k: v for k, v in r.items() if k != 'polygon_px'}, 'feature_type': 'building'},
        'geometry': mapping(Polygon(r['polygon_px']))
    })
for r in road_records:
    features.append({
        'type': 'Feature', 'properties': {'id': r['id'], 'length_m': r['length_m'], 'feature_type': 'road'},
        'geometry': mapping(LineString(r['polyline_px'])) if len(r['polyline_px']) >= 2 else None
    })
for r in water_records:
    features.append({
        'type': 'Feature', 'properties': {'id': r['id'], 'area_m2': r['area_m2'], 'feature_type': 'water'},
        'geometry': mapping(Polygon(r['polygon_px']))
    })

with open(f'{OUT_DIR}/all_features_final.geojson', 'w') as f:
    json.dump({'type': 'FeatureCollection', 'features': features}, f, indent=2)

with open(f'{OUT_DIR}/buildings_final.json', 'w') as f:
    json.dump(building_records, f, indent=2)
with open(f'{OUT_DIR}/roads_final.json', 'w') as f:
    json.dump(road_records, f, indent=2)
with open(f'{OUT_DIR}/water_final.json', 'w') as f:
    json.dump(water_records, f, indent=2)

print('\nAll features exported to all_features_final.geojson')
print(f'Summary: {len(building_records)} buildings, {len(road_records)} road segments, {len(water_records)} waterbodies')
