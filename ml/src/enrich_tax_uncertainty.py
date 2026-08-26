"""
Enrichment pass over the v2 building records - adds two capabilities named
in the tech stack but not yet implemented:

  1. PROPERTY TAX ESTIMATION - built-up area x roof-material valuation
     factor x a per-sqm base rate, mirroring how Urban/Gram Panchayat Local
     Bodies actually structure property tax slabs (construction-type-based
     multiplier on a per-area rate). Rates here are illustrative
     placeholders - a production system would pull the real per-district
     rate schedule from the relevant state Panchayati Raj / municipal API.

  2. UNCERTAINTY-AWARE REVIEW FLAGGING - rather than presenting every
     detection as equally confident, this reads back the model's raw
     probability map (pred_prob_v2.npy, pre-thresholding) and computes the
     mean predicted probability *inside* each building polygon. Buildings
     whose internal probability is close to the 0.5 decision boundary are
     flagged for human review instead of being silently trusted - this is
     the actual mechanism a real active-learning / human-in-the-loop
     pipeline would use to decide what to send to a reviewer queue.
"""
import json
import numpy as np
import cv2

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')

with open(f'{OUT_DIR}/buildings_final.json') as f:
    buildings = json.load(f)

pred_prob = np.load(f'{OUT_DIR}/pred_prob_final.npy')
H, W = pred_prob.shape

# ---- Property tax valuation table (illustrative - would come from a
# real per-district ULB/Panchayat schedule in production) ----
# Rate = annual tax in INR per m^2 of built-up area, varies by construction
# type since permanent (RCC) structures are taxed at a higher slab than
# temporary (tin-sheet) ones in most Indian municipal tax schedules.
TAX_RATE_PER_M2 = {
    'RCC (Concrete)': 45.0,
    'Tiled': 28.0,
    'Tin/Metal Sheet': 18.0,
    'Other': 15.0,
}


def uncertainty_flag(mean_prob):
    """Classify review priority based on distance from the 0.5 decision
    boundary - the actual signal a human-in-the-loop system would use."""
    dist = abs(mean_prob - 0.5)
    if dist < 0.10:
        return 'HIGH - needs review', round(float(mean_prob), 3)
    elif dist < 0.20:
        return 'MEDIUM - spot check', round(float(mean_prob), 3)
    else:
        return 'LOW - auto-accept', round(float(mean_prob), 3)


n_flagged_high = 0
n_flagged_medium = 0
total_tax = 0.0

for b in buildings:
    # --- property tax ---
    rate = TAX_RATE_PER_M2.get(b['roof_material'], 15.0)
    annual_tax = round(b['area_m2'] * rate, 2)
    b['estimated_annual_tax_inr'] = annual_tax
    total_tax += annual_tax

    # --- uncertainty flag: sample the probability map inside the polygon ---
    poly = np.array(b['polygon_px'], dtype=np.int32)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [poly], 1)
    px_probs = pred_prob[mask == 1]
    mean_prob = float(px_probs.mean()) if px_probs.size > 0 else 0.5

    review_flag, conf_val = uncertainty_flag(mean_prob)
    b['review_priority'] = review_flag
    b['mean_pixel_probability'] = conf_val

    if review_flag.startswith('HIGH'):
        n_flagged_high += 1
    elif review_flag.startswith('MEDIUM'):
        n_flagged_medium += 1

with open(f'{OUT_DIR}/buildings_final_enriched.json', 'w') as f:
    json.dump(buildings, f, indent=2)

print(f'Enriched {len(buildings)} buildings with tax + uncertainty data')
print(f'Total estimated annual property tax (this tile): Rs. {total_tax:,.0f}')
print(f'Review queue: {n_flagged_high} HIGH priority, {n_flagged_medium} MEDIUM priority '
      f'({(n_flagged_high+n_flagged_medium)/len(buildings)*100:.0f}% of buildings need some human review)')
print(f'Auto-accepted (low uncertainty): {len(buildings) - n_flagged_high - n_flagged_medium} buildings '
      f'({(len(buildings)-n_flagged_high-n_flagged_medium)/len(buildings)*100:.0f}%)')
