"""
Batch-process the 4 survey images with road_detector v3 and write improved outputs.

Usage:
    python process_survey_v3.py

Outputs to ml/outputs/survey_v3/:
 - per-area road JSON, overlay JPG, mask PNG
 - combined gallery comparison (orig vs v2 vs v3)
 - summary metrics printed
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
from skimage.morphology import skeletonize
from skimage.measure import label as cc_label, regionprops
from road_detector import detect_roads, vectorize_skeleton

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
SURVEY_DIR = os.path.join(REPO_ROOT, "dashboard", "public", "survey")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "survey_v3")
os.makedirs(OUT_DIR, exist_ok=True)

GSD_M = 0.5
FILES = ["area-01.jpg","area-02.jpg","area-03.jpg","area-04.jpg"]

# original/v2 helpers for side-by-side
def run_original(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    sat=hsv[:,:,1]; val=hsv[:,:,2]
    cand=((sat<60)&(val>70)&(val<200)).astype(np.uint8)
    LENGTH=21
    responses=np.zeros_like(gray,dtype=np.float32)
    for ang in range(0,180,15):
        se=np.zeros((LENGTH,LENGTH),np.uint8)
        cv2.line(se,(0,LENGTH//2),(LENGTH-1,LENGTH//2),1,1)
        M=cv2.getRotationMatrix2D((LENGTH/2,LENGTH/2),ang,1.0)
        se=cv2.warpAffine(se,M,(LENGTH,LENGTH),flags=cv2.INTER_NEAREST)
        responses=np.maximum(responses,cv2.morphologyEx(cand*255,cv2.MORPH_OPEN,se).astype(np.float32))
    road_mask=(responses>0).astype(np.uint8)*255
    road_mask=cv2.morphologyEx(road_mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    labeled=cc_label(road_mask>0)
    filtered=np.zeros_like(road_mask)
    for r in regionprops(labeled):
        if r.area<60: continue
        minr,minc,maxr,maxc=r.bbox
        h,w=maxr-minr,maxc-minc
        elong=max(h,w)/(min(h,w)+1e-6)
        if elong>2.0: filtered[labeled==r.label]=255
    skel=skeletonize(filtered>0)
    return filtered, skel

def run_v2(rgb):
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    sat=hsv[:,:,1]; val=hsv[:,:,2]; hue=hsv[:,:,0]
    asphalt=((sat<55)&(val>65)&(val<210)).astype(np.uint8)
    dirt=((hue>5)&(hue<25)&(sat>15)&(sat<80)&(val>80)&(val<190)).astype(np.uint8)
    concrete=((sat<35)&(val>=200)).astype(np.uint8)
    cand=np.clip(asphalt+dirt+concrete,0,1).astype(np.uint8)
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    LENGTH=31
    responses=np.zeros_like(gray,dtype=np.float32)
    for ang in range(0,180,10):
        se=np.zeros((LENGTH,LENGTH),np.uint8)
        cv2.line(se,(0,LENGTH//2),(LENGTH-1,LENGTH//2),1,1)
        M=cv2.getRotationMatrix2D((LENGTH/2,LENGTH/2),ang,1.0)
        se=cv2.warpAffine(se,M,(LENGTH,LENGTH),flags=cv2.INTER_NEAREST)
        responses=np.maximum(responses,cv2.morphologyEx(cand*255,cv2.MORPH_OPEN,se).astype(np.float32))
    road_mask=(responses>0).astype(np.uint8)*255
    road_mask=cv2.morphologyEx(road_mask,cv2.MORPH_CLOSE,np.ones((11,11),np.uint8))
    labeled=cc_label(road_mask>0)
    filtered=np.zeros_like(road_mask)
    for r in regionprops(labeled):
        if r.area<50: continue
        elong=r.axis_major_length/(r.axis_minor_length+1e-6)
        if elong>1.8: filtered[labeled==r.label]=255
    filtered=cv2.morphologyEx(filtered,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    skel=skeletonize(filtered>0)
    return filtered, skel

all_records={}
total_len_v3=0

print("="*70)
print("Survey V3 road processing (GSD 0.5m)")
print("="*70)

for fname in FILES:
    fpath=os.path.join(SURVEY_DIR,fname)
    if not os.path.exists(fpath):
        print(f"SKIP {fname} not found"); continue
    rgb=np.array(Image.open(fpath).convert("RGB"))
    H,W=rgb.shape[:2]
    stem=os.path.splitext(fname)[0]

    t0=time.time()
    _, filtered_v3, skel_v3, kept, blobs = detect_roads(rgb, GSD_M=GSD_M)
    records = vectorize_skeleton(skel_v3, GSD_M=GSD_M)
    all_records[stem]=records
    total_len_v3+=sum(r["length_m"] for r in records)
    print(f"\n[{fname}] v3: {len(records)} segments, {sum(r['length_m'] for r in records):.0f}m, kept {kept} blobs {blobs} time {time.time()-t0:.2f}s")

    # overlays
    overlay = rgb.copy()
    for rec in records:
        pts=np.array(rec["polyline_px"]).astype(int)
        for j in range(len(pts)-1):
            cv2.line(overlay, tuple(pts[j]), tuple(pts[j+1]), (255,220,0), 3)
    Image.fromarray(overlay).save(os.path.join(OUT_DIR, f"{stem}_v3_overlay.jpg"), quality=92)
    Image.fromarray(filtered_v3).save(os.path.join(OUT_DIR, f"{stem}_v3_mask.png"))

    # also save comparison triptych: original / v2 / v3
    filt_o, skel_o = run_original(rgb)
    filt_v2, skel_v2 = run_v2(rgb)
    def make_overlay(base, skel, color):
        out=base.copy()
        ys,xs=np.where(skel)
        # mask
        mask = (filtered_v3>0) if color==(80,255,80) else (filt_o>0 if color==(255,80,80) else filt_v2>0)
        # but simpler: just skeleton color
        out[ys,xs]=color
        return out
    # Create 3 panels with masks
    def overlay_with_mask(base, filt, skel, col):
        out=base.copy()
        m=filt>0
        out[m]=(out[m].astype(np.float32)*0.6 + np.array([255,220,80],np.float32)*0.4).astype(np.uint8)
        ys,xs=np.where(skel)
        out[ys,xs]=col
        return out
    vis_o=overlay_with_mask(rgb, filt_o, skel_o, (255,60,60))
    vis_v2=overlay_with_mask(rgb, filt_v2, skel_v2, (80,200,255))
    vis_v3=overlay_with_mask(rgb, filtered_v3, skel_v3, (60,255,100))
    for img,txt in [(vis_o,"ORIGINAL"),(vis_v2,"V2"),(vis_v3,"V3 (veg+width+bridge)")]:
        cv2.putText(img, txt, (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3)
        cv2.putText(img, txt, (15,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1)
    triptych=np.concatenate([vis_o,vis_v2,vis_v3],axis=1)
    Image.fromarray(triptych).save(os.path.join(OUT_DIR, f"{stem}_triptych.jpg"), quality=88)

    # per-area json
    with open(os.path.join(OUT_DIR, f"{stem}_roads_v3.json"),"w") as f:
        json.dump(records,f,indent=2)

# combined json
with open(os.path.join(OUT_DIR,"roads_survey_v3_all.json"),"w") as f:
    json.dump(all_records,f,indent=2)

print("\n"+"="*70)
print(f"Total v3 length across 4 images: {total_len_v3:.0f} m")
print(f"Outputs in: {OUT_DIR}")
for f in sorted(os.listdir(OUT_DIR)):
    print(" ",f)

# also produce a summary for dashboard: map survey areas to bbox-filtered roads
# The dashboard filters roads.json by bbox; we can generate a unified roads.json
# that contains all v3 vectors with coordinates in a global 1000x1000 space
# For survey images, coordinates are per-image 0-1000; for dashboard quadrants we could offset
# But simpler: just report metrics
