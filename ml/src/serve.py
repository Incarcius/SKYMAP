"""
FastAPI server for SKYMAP upload inference.

Run:
  python -m ml.src.serve
  # or
  uvicorn ml.src.serve:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
  POST /api/infer  - multipart file + gsd, returns buildings/roads/water + overlay b64
  GET  /api/health
  GET  /
"""
import os, sys, io, base64, json, time
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import numpy as np, cv2
from PIL import Image

from road_detector import detect_roads, vectorize_skeleton

app = FastAPI(title="SKYMAP Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# try building model once at startup (prefer SVAMITVA Indian villages model, then fallback to original)
building_model = None
building_ckpt_path = None
try:
    import torch
    try:
        import numpy.core.multiarray
        import numpy._core.multiarray
        torch.serialization.add_safe_globals([numpy._core.multiarray.scalar])
    except: pass
    from model import AttentionResUNet
    candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "building_model_svamitva.pt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "road_model_svamitva.pt"), # not building but keep
        os.path.join("/home/rigalis/SIH2027/svamitva_prototype/final_package/outputs/models/final_model_epoch27.pt"),
        os.path.join("/home/rigalis/SIH2027/SKYMAP/ml/outputs/model_v2_ckpt.pt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "model_v2_ckpt.pt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "model_v2_ckpt.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            building_ckpt_path = p
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
            except:
                ckpt = torch.load(p, map_location="cpu")
            m = AttentionResUNet(in_ch=3, out_ch=1, base=16)
            # handle if ckpt has different base (some saved with 32M) — try to load strictly, else skip
            try:
                m.load_state_dict(ckpt["model"])
            except Exception as e:
                print(f"[serve] Failed to load {p}: {e}")
                continue
            m.eval()
            building_model = m
            print(f"[serve] Loaded building checkpoint: {p} epoch {ckpt.get('epoch','?')} IoU {ckpt.get('iou','?')}")
            break
    if building_model is None:
        print("[serve] No building checkpoint found — roads+water only (still useful for upload).")
except Exception as e:
    print(f"[serve] Building model not loaded: {e}")
    import traceback; traceback.print_exc()
    building_model = None

# road learned model (optional, for Indian villages)
road_learned_model = None
try:
    import torch
    from model import AttentionResUNet as RoadUNet
    road_ckpt_candidates = [
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "road_model_svamitva.pt"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "road_model_ckpt.pt"),
    ]
    for p in road_ckpt_candidates:
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
            except:
                ckpt = torch.load(p, map_location="cpu")
            rm = RoadUNet(in_ch=3, out_ch=1, base=16)
            try:
                rm.load_state_dict(ckpt["model"])
                rm.eval()
                road_learned_model = rm
                print(f"[serve] Loaded ROAD learned checkpoint: {p} epoch {ckpt.get('epoch')} IoU {ckpt.get('iou')}")
                break
            except: pass
except Exception as e:
    print(f"[serve] Road learned model not loaded, using classical V3: {e}")
    road_learned_model = None

def infer_buildings(rgb_u8, gsd=0.5):
    if building_model is None:
        return [], None
    import torch
    H,W = rgb_u8.shape[:2]
    rgb = rgb_u8.astype(np.float32)/255.0
    patch, stride = 128, 96
    pred_sum = np.zeros((H,W), np.float32)
    pred_cnt = np.zeros((H,W), np.float32)
    ys = sorted(set(list(range(0, H-patch+1, stride)) + [H-patch]))
    xs = sorted(set(list(range(0, W-patch+1, stride)) + [W-patch]))
    with torch.no_grad():
        for y in ys:
            for x in xs:
                tile = rgb[y:y+patch, x:x+patch]
                t = torch.from_numpy(tile.transpose(2,0,1)).float().unsqueeze(0)
                out = torch.sigmoid(building_model(t)).squeeze().numpy()
                pred_sum[y:y+patch, x:x+patch] += out
                pred_cnt[y:y+patch, x:x+patch] += 1
    pred_cnt[pred_cnt==0]=1
    prob = pred_sum/pred_cnt
    # use tuned threshold if available else 0.5
    thresh = 0.5
    try:
        # try ensemble threshold
        tp = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "ensemble_threshold.txt")
        if os.path.exists(tp):
            thresh = float(open(tp).read().strip())
    except: pass
    mask = (prob > thresh).astype(np.uint8)*255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
    contours,_ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    records=[]
    for i,c in enumerate(contours):
        area=cv2.contourArea(c)
        if area<40: continue
        eps=0.01*cv2.arcLength(c,True)
        approx=cv2.approxPolyDP(c,eps,True)
        coords=[[int(p[0]),int(p[1])] for p in approx.reshape(-1,2)]
        if len(coords)<3: continue
        records.append({"id":int(i),"polygon_px":coords,"area_m2":float(round(area*(gsd**2),1)),"bbox":[int(cv2.boundingRect(c)[0]),int(cv2.boundingRect(c)[1]),int(cv2.boundingRect(c)[2]),int(cv2.boundingRect(c)[3])]})
    return records, clean

@app.get("/api/health")
def health():
    return {"status":"ok","building_model": building_model is not None, "road_detector":"v3"}

@app.post("/api/infer")
async def infer(file: UploadFile = File(...), gsd: float = Form(0.5)):
    t0=time.time()
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error":f"Invalid image: {e}"}, status_code=400)
    rgb = np.array(img)
    H,W = rgb.shape[:2]
    if max(H,W) > 3000:
        # downscale large uploads to avoid OOM, keep aspect
        scale = 3000 / max(H,W)
        new_w, new_h = int(W*scale), int(H*scale)
        img = img.resize((new_w,new_h), Image.LANCZOS)
        rgb = np.array(img)
        H,W = rgb.shape[:2]
    # buildings
    building_records, building_mask = infer_buildings(rgb, gsd)
    # roads v3
    _, road_filtered, skeleton, kept, blobs = detect_roads(rgb, building_clean=building_mask, GSD_M=gsd)
    road_records = vectorize_skeleton(skeleton, GSD_M=gsd)
    # water
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
        water_records.append({"id":int(i),"polygon_px":coords,"area_m2":float(round(area*(gsd**2),1))})
    # overlay for preview (buildings red, roads yellow, water cyan)
    overlay=rgb.copy()
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
    # encode overlay to b64 jpeg
    pil_overlay=Image.fromarray(overlay)
    buf=io.BytesIO()
    pil_overlay.save(buf, format="JPEG", quality=88)
    overlay_b64=base64.b64encode(buf.getvalue()).decode()
    # also encode input as b64 for map display
    buf2=io.BytesIO()
    img.save(buf2, format="JPEG", quality=90)
    input_b64=base64.b64encode(buf2.getvalue()).decode()
    elapsed=time.time()-t0
    return {
        "width":W,"height":H,"gsd":gsd,
        "elapsed_sec": round(elapsed,2),
        "counts":{"buildings":len(building_records),"roads":len(road_records),"water":len(water_records),
                  "road_kept":kept,"road_blobs":blobs,
                  "total_road_m": round(sum(r["length_m"] for r in road_records),1)},
        "buildings": building_records,
        "roads": road_records,
        "water": water_records,
        "overlay_b64": overlay_b64,
        "input_b64": input_b64,
    }

@app.get("/")
def root():
    return {"message":"SKYMAP Inference API — POST /api/infer with file + gsd"}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("serve:app", host="0.0.0.0", port=8000, reload=True)
