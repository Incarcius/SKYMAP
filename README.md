# SKYMAP — AI Drone Feature Extraction & GCS Dashboard

**SKYMAP** is an end-to-end AI/ML drone feature extraction solution and UAV Ground Control Station (GCS) dashboard. It processes high-resolution drone orthophotos to extract building footprints (with roof-material classification), road networks, and waterbodies, while generating property tax estimates, solar potential ratings, and change detection overlays.

---

## 📁 Repository Structure

```
sih26/
├── README.md                      <- Root repository documentation
├── MANUAL.md                      <- Detailed execution & testing manual
├── PROJECT_OVERVIEW.md            <- ML architecture, methodologies, and metrics report
├── dashboard_best.html            <- Standalone self-contained HTML dashboard viewer
├── ml-prototype/                  <- PyTorch ML model, feature vectorization & pipeline scripts
│   ├── code/                      <- Python training, inference & pipeline scripts
│   ├── data/                      <- RGB orthophoto source tile and ground-truth mask
│   └── outputs/                   <- Generated JSONs, GeoJSONs, models, & demo galleries
└── gcs-dashboard/                 <- React + Vite Ground Control Station web application
    ├── src/                       <- React components (FlightMap, TelemetryHUD, Sidebars, Modals)
    ├── public/                    <- Static assets and ML demonstration gallery
    └── gen_b64.py                 <- Utility script to embed background orthophoto
```

---

## 🚀 Quick Start

### 1. React Ground Control Station (GCS Dashboard)
```bash
cd gcs-dashboard
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) to launch the interactive SKYMAP GCS Dashboard.

### 2. ML Pipeline & Model Training
```bash
cd ml-prototype
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run full feature extraction pipeline:
python3 code/run_full_pipeline.py
```

### 3. Standalone Dashboard Preview
Open `dashboard_best.html` directly in any web browser without needing a web server or dependencies.

---

## 📊 Feature Coverage & ML Highlights

- **Building Footprints**: Attention-ResUNet + ASPP model with 3-model ensembling and Test-Time Augmentation (IoU 0.401).
- **Polygon Regularization**: Snap-to-rectilinear cleanup and edge straightening for CAD/GIS export.
- **Roof Classification**: KMeans unsupervised cluster-based classification (RCC / Tiled / Tin / Other).
- **Road & Water Extraction**: Morphological multi-angle linear filter and HSV-texture thresholding.
- **Derived Analytics**: Automated property tax estimation, solar capacity (kWp), and uncertainty review flagging.
- **ML Gallery & Stress Testing**: Built-in visual gallery and domain-shift stress testing viewer in the dashboard.
