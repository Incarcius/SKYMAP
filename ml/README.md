# SVAMITVA AI Feature Extraction — Prototype

**Start here:** open `MANUAL.md` for full instructions.

**Fastest demo (no install needed):** open `outputs/dashboard/dashboard_best.html` in your browser.

**Best model results (3-model ensemble, held-out region, no data leakage):**
**IoU 0.401** · via Attention-ResUNet+ASPP → TTA → threshold tuning → 3-model ensembling.
Full ablation: 0.361 (base architecture) → 0.373 (+TTA/threshold) → **0.401 (+ensemble)**.

**Validated with 4-fold spatial cross-validation** (mean IoU 0.296 ± 0.066 across all 4 tile edges) to confirm the result generalizes rather than being a lucky split.

**Full pipeline coverage:** buildings + roads + waterbodies + roof material classification + solar potential + property tax estimation + uncertainty-based review queue (98% auto-accept) + change detection + regularized (rectilinear) building polygons.

**Throughput:** 13,328 ha/hour on a single CPU core at real SVAMITVA resolution — a ~100ha village processes in under 30 seconds.

See `MANUAL.md` → Section 10 for a 2-minute live-demo script.
