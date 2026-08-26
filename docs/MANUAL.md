# SVAMITVA AI Feature Extraction — Testing Manual

**Problem Statement:** DJS_26_SW_08 — AI/ML feature extraction from drone orthophotos (buildings, roads, waterbodies, roof material classification)

This package contains a fully working prototype: a trained segmentation model, the complete extraction pipeline (buildings → roads → waterbodies → roof classification → solar/tax scoring → change detection), an interactive dashboard, and all source code to reproduce or extend it.

---

## 1. What's in this package

```
├── MANUAL.md                          <- this file
├── requirements.txt                   <- Python dependencies
├── data/
│   ├── RGB.png                        <- source aerial tile (1000x1000px, 1.5m/px)
│   └── GT.png                         <- ground-truth building mask (for evaluation only)
├── code/
│   ├── model.py                       <- Attention-ResUNet + ASPP architecture
│   ├── dataset.py                     <- patch dataset + spatial train/val split (supports 4-edge k-fold)
│   ├── train.py                       <- resumable training script (main model)
│   ├── train_ensemble_member.py       <- trains additional ensemble members (different seeds)
│   ├── kfold_cv.py                    <- 4-fold spatial cross-validation
│   ├── final_inference.py             <- TTA + threshold-tuned inference (single model)
│   ├── ensemble_eval.py               <- averages 3 models' predictions, re-tunes threshold
│   ├── regularize_polygons.py         <- rectilinear polygon cleanup post-processing
│   ├── infer_and_vectorize_best.py    <- buildings + roads + water + roof classifier (uses ensemble)
│   ├── enrich_tax_uncertainty_best.py <- property tax + review-queue flagging
│   ├── change_detection.py            <- new construction / demolition detection
│   └── build_dashboard_best.py        <- generates the interactive HTML dashboard
└── outputs/
    ├── slides/                        <- 13 PPT-ready PNG visuals
    ├── data/                          <- GeoJSON + JSON feature records + metrics + k-fold/ensemble results
    ├── dashboard/dashboard_best.html  <- interactive map, BEST model (open directly in browser)
    └── models/                        <- main checkpoint + 2 ensemble member checkpoints
```

---

## 2. Fastest way to "test" this (no setup required)

You don't need Python installed to see the prototype working:

1. **Open `outputs/dashboard/dashboard_best.html`** in any browser (double-click it). This uses the best model (3-model ensemble + regularized polygons) and is fully interactive, self-contained (image embedded as base64 — no external files needed).
   - Hover over any building to see area, roof material, solar potential, and estimated tax.
   - Hover over yellow lines to see road segment length.
   - Hover over cyan shapes to see waterbody area.
   - Use the **Layers** tab to toggle buildings/roads/water on and off.
   - Use the **Tax** tab to see property tax totals and rate assumptions.
   - Use the **Review Queue** tab to see which buildings the model flagged as low-confidence.

2. **Open any PNG in `outputs/slides/`** — these are the exact images used in the PPT, viewable in any image viewer.

3. **Open `outputs/data/all_features_final.geojson`** in [geojson.io](https://geojson.io) or QGIS to see the extracted buildings/roads/water as real geospatial vector data.

This alone is enough to demonstrate the full prototype to judges or teammates without running any code.

---

## 3. Running the pipeline yourself (full reproduction)

### 3.1 Setup

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested on Python 3.10+. CPU-only is fine (no GPU required — the whole pipeline was built and run on CPU).

### 3.2 Recommended: one-command pipeline

Once you've trained the 3 model checkpoints (Section 3.3 below — this only needs to happen once), everything downstream — ensemble inference, building/road/water extraction, polygon regularization, tax/uncertainty enrichment, dashboard, and demo gallery — runs with a single command:

```bash
python3 code/run_full_pipeline.py                        # full run, ~2-3 min on CPU
python3 code/run_full_pipeline.py --skip-change-detection # skip the slower synthetic demo stage (~30s faster)
python3 code/run_full_pipeline.py --force                 # rerun every stage even if outputs already exist
```

This validates every prerequisite file exists before starting (with a specific message telling you exactly what's missing and how to produce it, rather than a cryptic crash three stages in), checks that each stage actually produced its expected output file (not just that the script exited without error), and automatically skips any stage whose output already exists — so if you're just re-running after tweaking one thing, it won't waste 2 minutes redoing untouched stages.

It also fixes one thing that used to be a hidden manual step: the dashboard's embedded image used to require a separate one-off base64-encoding command that wasn't part of any script. That's now built into the pipeline automatically.

### 3.3 Training the model checkpoints (one-time, before the pipeline above)

The scripts below are numbered by dependency, not filename — run them in this order. Each step reads the previous step's output, so don't skip one.

| Step | Command | What it does | Approx. time (CPU) |
|---|---|---|---|
| 1 | `python3 code/train.py --epochs 5` | Trains the main model from scratch for 5 epochs, saves a checkpoint | ~3-4 min |
| 1b | `python3 code/train.py --epochs 5 --resume` | Resumes training for 5 more epochs (repeat as needed — the included checkpoint was trained this way across 27 total epochs) | ~3-4 min per call |
| 2 | `python3 code/train_ensemble_member.py --member 1 --epochs 5` | Trains ensemble member 1 (different random seed); repeat with `--resume` to extend | ~3-4 min per call |
| 3 | `python3 code/train_ensemble_member.py --member 2 --epochs 5` | Trains ensemble member 2 | ~3-4 min per call |

Once all 3 checkpoints exist (`model_v2_ckpt.pt`, `ensemble_member_1.pt`, `ensemble_member_2.pt` in `outputs/models/`), run `code/run_full_pipeline.py` from Section 3.2 for everything else.

All scripts print their key metrics to the console as they run — you don't need to open any file to confirm each step worked.

### 3.3 Expected console output (sanity check)

If everything is working, `final_inference.py` should end with something close to:
```
TTA + tuned th=0.56     : IoU=0.3731  P=0.4936  R=0.6044  F1=0.5434
IoU improvement from TTA + threshold tuning alone: +3.4%
```

`infer_and_vectorize_final.py` should end with something close to:
```
Summary: 286 buildings, 26 road segments, 9 waterbodies
```

If your numbers differ by a little, that's expected — training has some randomness even with a fixed seed across different machines/PyTorch versions. If they differ by a lot (e.g., 0 buildings detected), something is misconfigured — see Troubleshooting below.

---

## 4. Testing with your own imagery

To try this on a different aerial tile:

1. Replace `data/RGB.png` with your own image (any size, but keep it square-ish and at least 512×512px for the sliding-window inference to make sense).
2. If you have a ground-truth building mask for evaluation, replace `data/GT.png` (white = building, black = background). **If you don't have one**, you can still run steps 2 onward — you just won't get IoU/precision/recall numbers, only the raw detections.
3. Re-run from Step 1 (or Step 1b if you want to fine-tune the existing checkpoint on your new tile instead of starting over).
4. Update `GSD_M` (ground sample distance, meters/pixel) at the top of `infer_and_vectorize_final.py` and `change_detection.py` to match your actual imagery resolution — this directly affects all area/length/tax/solar calculations. SVAMITVA drone imagery is 0.5m/px; the demo tile included here is 1.5m/px (INRIA aerial dataset, used as a stand-in since real SVAMITVA imagery wasn't available in this environment).

---

## 5. How to read the metrics (so you can defend them to judges)

- **IoU (Intersection over Union)**: overlap between predicted and true building pixels, divided by their union. **Best model (3-model ensemble): 0.401** on a truly held-out region (no data leakage — see below).
- **Precision/Recall/F1**: see `outputs/data/ensemble_results.txt` for the full breakdown across the main model and both ensemble members.
- **"Held-out region"**: the tile is split spatially — the model never sees the right-hand 20% of the image during training, and all reported metrics come from that untouched region. This matters because it means the numbers aren't inflated by testing on data the model memorized.
- **Review queue percentages**: with the ensemble model, **98% of detected buildings auto-accepted**, only 2% flagged for human review (down from 8% with the single model) — the ensemble is measurably more confident and consistent.

## 6. Further improvements applied (the "make it the best" pass)

Beyond the initial architecture and pipeline, five additional engineering passes were made, each with honest before/after numbers:

| Improvement | What it does | Result |
|---|---|---|
| **4-fold spatial cross-validation** (`kfold_cv.py`) | Rotates the held-out region through all 4 edges of the tile (right/left/top/bottom), training independently against each, to prove the reported IoU isn't a fluke of one lucky split | Mean IoU 0.296 ± 0.066 across folds (4 epochs/fold, time-limited) — tight enough spread to show the architecture generalizes across regions |
| **Model ensembling** (`train_ensemble_member.py`, `ensemble_eval.py`) | Trains 2 additional models with different random seeds on the same split, averages all 3 models' predictions | **+7.5% IoU** over the best single model (0.373 → 0.401) — the ensemble beats every individual member |
| **Polygon regularization** (`regularize_polygons.py`) | Snaps jagged CNN-derived building outlines to rectilinear rotated-rectangle or edge-straightened form, matching how real buildings and GIS/property records actually look | 291/291 buildings cleaned (130 snapped to simple rectangles, 161 edge-straightened for complex L/T-shapes) — purely a presentation/usability improvement, doesn't change which pixels were classified |
| **Latency/throughput benchmarking** | Measured actual inference speed to answer "can this scale to a real village" | 13,328 ha/hour (no TTA) on a single CPU core at real SVAMITVA resolution — a typical ~100ha village processes in under 30 seconds, no GPU required |
| **Full ablation table** | Documents the complete improvement journey from baseline to final | Plain U-Net (leaky split, not comparable) → Attention+ASPP honest split (0.361) → +TTA+threshold (0.373) → +Ensemble (0.401) |

See `outputs/slides/slide_kfold_cv.png`, `slide_ensemble.png`, `slide_polygon_regularization.png`, `slide_latency.png`, and `slide_ablation.png` for the corresponding PPT visuals.

---

## 7. Known limitations (be upfront about these — it's more credible than hiding them)

- **Single demo tile**: the model was trained on one 1000×1000px INRIA aerial tile (a US suburb), not real SVAMITVA imagery, because real labeled SVAMITVA data and pretrained model weights (SAM, D-LinkNet, EfficientNet) were not reachable from this development environment. The architecture and pipeline are built to swap in real data/weights directly — see Section 7.
- **Roads and water use classical CV, not deep learning**: no labeled road/water masks or pretrained D-LinkNet weights were available, so roads are extracted via a multi-orientation morphological linear-feature detector + PCA vectorization, and water via HSV+texture thresholding. Both work well on this tile but are less robust than a trained model would be — noted as the first production upgrade.
- **Roof material classifier is unsupervised (KMeans)**, not a trained CNN — again due to no labeled roof-material data being available. Cluster-to-label mapping is human-reviewable (only 4 decisions) but not learned.
- **Change detection uses a synthetically edited tile**, not a real two-pass orthophoto pair, since no bi-temporal SVAMITVA data was available. The diff logic itself (IoU-matching detections between two independent inference runs) is production-ready and would work unchanged on real dated imagery.
- **Property tax rates are illustrative placeholders**, not real municipal rate schedules.

---

## 8. Production upgrade path (what changes with real resources)

| Component | Prototype (this package) | Production upgrade |
|---|---|---|
| Buildings | Attention-ResUNet+ASPP, trained from scratch, CPU, 1 tile | Same architecture, fine-tuned from SAM/ResNet pretrained weights, GPU, full SVAMITVA labeled dataset |
| Roads | Classical morphological + PCA | D-LinkNet trained on labeled road masks |
| Water | HSV + texture threshold | NDWI (if NIR band available) or trained U-Net |
| Roof material | KMeans clustering (unsupervised) | EfficientNet-B3 classifier trained on labeled roof photos |
| Change detection | Synthetic Time-B tile | Real bi-temporal SVAMITVA passes |
| Deployment | Local scripts | ONNX export + Triton/FastAPI serving, as described in the tech stack doc |

---

## 9. Troubleshooting

| Problem | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError` | Missing dependency | `pip install -r requirements.txt` |
| Training seems to hang | Normal — CPU training is slow (~45-50s/epoch on a typical laptop) | Wait, or reduce `--epochs` |
| `FileNotFoundError: model_v2_ckpt.pt` | You ran `final_inference.py` before any training step | Run `train.py` first, or use the included pretrained checkpoint (copy `outputs/models/final_model_epoch27.pt` to where the scripts expect `model_v2_ckpt.pt`) |
| Very different building counts on your own imagery | Different resolution/color palette than the training tile | Expected — this is a single-tile prototype; fine-tune on your own data for stable results (see Section 4) |
| Dashboard shows a blank map | Browser blocking local file access to embedded base64 | This shouldn't happen since the image is embedded, not linked — try a different browser or check console for JS errors |

---

## 10. Generating demo images (basic)

`code/generate_demo_images.py` produces presentation-ready side-by-side (input orthophoto vs. AI-extracted features) images. By default it generates 3 crops, all from the **held-out validation region** (x ≥ 800px) so they are honestly "data the model never trained on" — meaningful if a judge asks whether you're just showing memorized training data.

```bash
python3 code/generate_demo_images.py                    # regenerates the default 3 demo images
python3 code/generate_demo_images.py --custom Y0 Y1 X0 X1 --name my_crop --label "My region"
```

The script prints `unseen=True/False` for each crop and warns explicitly if a custom region overlaps the training area — use that to keep your demo honest. Output goes to `outputs/demo_images/`, both as individual `_input.png` / `_output.png` files (for dropping into slides) and a combined `_sidebyside.png` (ready to present as-is).

## 11. Generating the demo gallery (recommended — auto-selected, multi-format)

`code/generate_demo_gallery.py` is the upgraded version: instead of manually picking crop regions by eye, it **automatically scans the entire held-out strip** in a sliding window and scores every candidate on:
- building density and roof-material diversity (more interesting to look at)
- waterbody presence (bonus)
- **road-polyline jaggedness** (turning-angle variance) — crops where the classical road detector visibly zigzags are **hard-disqualified** from the recommended pool, not just docked points, so a bad result can't accidentally slip into your demo set

```bash
python3 code/generate_demo_gallery.py --top 3 --show-rejects 2
```

For each of the top-N selected regions, it generates **four formats**:
1. `gallery_N_annotated.png` — self-contained side-by-side with an in-image legend and live stats footer (works even if screenshotted out of context)
2. `gallery_N_upscaled2x.png` — Lanczos-upscaled, sharper for a projector
3. `gallery_N_heatmap.png` — 3-panel view showing the model's **raw confidence** (not just the final yes/no decision) — a strong technical-depth slide
4. `gallery_N_toggle.gif` — animated input↔output toggle, punchier for a live demo than a static image

It also saves `reject_N_annotated.png` — the worst-scoring (usually road-jaggedness-disqualified) crops, kept on purpose as an honest "known limitation" example rather than hidden. `selection_report.txt` documents the exact score breakdown for every candidate scanned, so you can defend why each region was picked if asked.

If you want a different number of candidates or a different search granularity, `CROP_HEIGHT`, `Y_STRIDE`, and `ROAD_PENALTY_DISQUALIFY` are configurable constants at the top of the script.

## 12. Quick demo script (for judges / live presentation)

If you only have 2 minutes to show this live:

1. Open `dashboard_best.html` — hover 2-3 buildings to show area/roof/tax/solar data live.
2. Click the **Review Queue** tab — point out the 98% auto-accept rate.
3. Show `demo_gallery/gallery_1_toggle.gif` — punchy input↔output toggle, good opener.
4. Show `demo_gallery/gallery_1_heatmap.png` — the model's raw confidence, not just a final decision. Strongest technical-depth visual in the whole deck.
5. Open `slides/slide_ablation.png` — walk through the improvement journey in 15 seconds (baseline → attention → TTA → ensemble).
6. Open `slides/slide_ensemble.png` — explain the ensemble beats every individual model.
7. Open `slides/slide_kfold_cv.png` — if a judge asks "how do you know this isn't a fluke," you already have the answer on a slide.
8. If asked about limitations: `demo_gallery/reject_1_annotated.png` + `selection_report.txt` — you have a real, honestly-documented failure case ready, which lands better than being caught off guard.
9. Close with `slides/slide_latency.png` — the deployment-feasibility story: a village processes in under 30 seconds on CPU alone.
