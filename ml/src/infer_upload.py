#!/usr/bin/env python3
"""
Upload inference — run full SKYMAP detection on any user image.

Usage:
  python ml/src/infer_upload.py /path/to/your_image.jpg [--gsd 0.5] [--out outputs/upload]

Output:
  upload_overlay.jpg  (buildings + roads + water drawn)
  upload_roads.json, upload_buildings.json (if checkpoint available)
  road_mask.png

This is the CLI backend for the dashboard upload feature.
Frontend upload just POSTs the file to this logic (see serve.py).
"""
import argparse, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, cv2
from PIL import Image
from road_detector import detect_roads, vectorize_skeleton

# try to load building model if checkpoint exists
def try_building_inference(rgb_u8, out_dir, gsd):
    try:
        import torch
        from model import AttentionResUNet
        ckpt_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "model_v2_ckpt.pt"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "SIH2027", "SKYMAP", "ml", "outputs", "model_v2_ckpt.pt"),
            os.path.join("/home/rigalis/SIH2027/SKYMAP/ml/outputs/model_v2_ckpt.pt"),
        ]
        ckpt_path = next((p for p in ckpt_paths if os.path.exists(p)), None)
        if ckpt_path is None:
            print("Building model: no checkpoint found, skipping buildings (roads+water only).")
            return [], None
        # simple tiled inference without TTA for arbitrary size
        from pathlib import Path
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = AttentionResUNet(in_ch=3, out_ch=1, base=16)
        model.load_state_dict(ckpt["model"])
        model.eval()
        H,W = rgb_u8.shape[:2]
        rgb = rgb_u8.astype(np.float32)/255.0
        # naive single-pass tiled
        patch=128; stride=96
        pred_sum=np.zeros((H,W),np.float32)
        pred_cnt=np.zeros((H,W),np.float32)
        ys=sorted(set(list(range(0,H-patch+1,stride))+[H-patch]))
        xs=sorted(set(list(range(0,W-patch+1,stride))+[W-patch]))
        import torch as _torch
        with _torch.no_grad():
            for y in ys:
                for x in xs:
                    tile=rgb[y:y+patch,x:x+patch]
                    t=_torch.from_numpy(tile.transpose(2,0,1)).float().unsqueeze(0)
                    out=_torch.sigmoid(model(t)).squeeze().numpy()
                    pred_sum[y:y+patch,x:x+patch]+=out
                    pred_cnt[y:y+patch,x:x+patch]+=1
        pred_cnt[pred_cnt==0]=1
        prob=pred_sum/pred_cnt
        # threshold 0.5
        mask=(prob>0.5).astype(np.uint8)*255
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        clean=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
        clean=cv2.morphologyEx(clean,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(clean,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        records=[]
        for i,c in enumerate(contours):
            area=cv2.contourArea(c)
            if area<40: continue
            eps=0.01*cv2.arcLength(c,True)
            approx=cv2.approxPolyDP(c,eps,True)
            coords=[[int(p[0]),int(p[1])] for p in approx.reshape(-1,2)]
            if len(coords)<3: continue
            records.append({"id":int(i),"polygon_px":coords,"area_m2":float(round(area*(gsd**2),1))})
        print(f"Buildings: {len(records)} detected (prob threshold 0.5)")
        return records, clean
    except Exception as e:
        print(f"Building inference skipped: {e}")
        return [], None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("image", help="path to RGB image (jpg/png)")
    ap.add_argument("--gsd", type=float, default=0.5, help="meters per pixel (0.5 for SVAMITVA)")
    ap.add_argument("--out", default="outputs/upload", help="output dir")
    args=ap.parse_args()
    if not os.path.exists(args.image):
        print(f"Not found: {args.image}"); sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    rgb=np.array(Image.open(args.image).convert("RGB"))
    print(f"Input: {args.image} {rgb.shape[1]}x{rgb.shape[0]} GSD={args.gsd}m")
    # buildings (optional)
    building_records, building_mask = try_building_inference(rgb, args.out, args.gsd)
    # roads v3
    _, road_filtered, skeleton, kept, blobs = detect_roads(rgb, building_clean=building_mask, GSD_M=args.gsd)
    road_records = vectorize_skeleton(skeleton, GSD_M=args.gsd)
    print(f"Roads: {len(road_records)} segments, {sum(r['length_m'] for r in road_records):.0f}m  (kept {kept} blobs {blobs})")
    # water (simple HSV)
    hsv=cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue,sat,val=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    gray=cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    water_candidate=(((hue>85)&(hue<135))&(sat>40)&(val>60)&(val<220)).astype(np.uint8)
    tex=cv2.Laplacian(gray,cv2.CV_64F)
    tex=cv2.GaussianBlur(np.abs(tex).astype(np.float32),(9,9),0)
    water_candidate[tex>15]=0
    if building_mask is not None:
        dil=cv2.dilate(building_mask,np.ones((9,9),np.uint8))
        water_candidate[dil>0]=0
    water_mask=cv2.morphologyEx(water_candidate*255,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    water_mask=cv2.morphologyEx(water_mask,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    w_contours,_=cv2.findContours(water_mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    water_records=[]
    for i,c in enumerate(w_contours):
        area=cv2.contourArea(c)
        if area<25: continue
        eps=0.01*cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,eps,True)
        coords=[[int(p[0]),int(p[1])] for p in approx.reshape(-1,2)]
        if len(coords)<3: continue
        water_records.append({"id":int(i),"polygon_px":coords,"area_m2":float(round(area*(args.gsd**2),1))})
    print(f"Water: {len(water_records)}")
    # overlay
    overlay=rgb.copy()
    # buildings red
    for rec in building_records:
        pts=np.array(rec["polygon_px"])
        cv2.polylines(overlay,[pts],True,(255,80,80),2)
    for rec in road_records:
        pts=np.array(rec["polyline_px"]).astype(int)
        for j in range(len(pts)-1):
            cv2.line(overlay,tuple(pts[j]),tuple(pts[j+1]),(255,220,0),3)
    for rec in water_records:
        pts=np.array(rec["polygon_px"])
        cv2.polylines(overlay,[pts],True,(0,220,220),2)
    Image.fromarray(overlay).save(os.path.join(args.out,"upload_overlay.jpg"), quality=92)
    Image.fromarray(road_filtered).save(os.path.join(args.out,"road_mask.png"))
    if building_mask is not None:
        Image.fromarray(building_mask).save(os.path.join(args.out,"building_mask.png"))
    Image.fromarray(water_mask).save(os.path.join(args.out,"water_mask.png"))
    with open(os.path.join(args.out,"upload_roads.json"),"w") as f: json.dump(road_records,f,indent=2)
    with open(os.path.join(args.out,"upload_buildings.json"),"w") as f: json.dump(building_records,f,indent=2)
    with open(os.path.join(args.out,"upload_water.json"),"w") as f: json.dump(water_records,f,indent=2)
    print(f"\nSaved to {args.out}/")
    print("  upload_overlay.jpg  <- open this")
    print("  upload_roads.json / upload_buildings.json")

if __name__=="__main__":
    main()
