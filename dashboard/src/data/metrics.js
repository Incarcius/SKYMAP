// src/data/metrics.js
//
// Model-performance numbers for the Overall Metrics Page.
// Source: docs/PROJECT_OVERVIEW.md (Sections 5.1–5.6) and docs/MANUAL.md.
// These are the project's actual benchmark results — not placeholders.
// Only the Human-in-the-Loop section is left to be computed live from
// buildings.json (see utils/systemStatus.js:computeReviewStats) so it never
// goes stale relative to the data actually loaded in the app.

export const buildingSegmentation = {
  iou: 40.1,
  precision: 53.9,
  recall: 61.0,
  f1: 57.2,
};

export const improvementJourney = [
  { stage: 'v1: Plain U-Net', iou: 28.1, note: 'Leakage bug — not comparable' },
  { stage: 'v2: Attention-ResUNet+ASPP', iou: 36.1, note: 'First honest, leakage-free number' },
  { stage: 'v2 + TTA + tuned threshold', iou: 37.3, note: '+3.4% IoU, zero extra training' },
  { stage: 'Final: 3-model ensemble', iou: 40.1, note: 'Best result — +7.5% over best single model' },
];

export const ensembleComparison = {
  members: [
    { label: 'Main Model (seed 42, 27 epochs)', iou: 37.3 },
    { label: 'Member 1 (seed 201, 18 epochs)', iou: 37.6 },
    { label: 'Member 2 (seed 202, 15 epochs)', iou: 34.1 },
  ],
  ensemble: { label: 'Ensemble (3-model average)', iou: 40.1 },
  relativeImprovementPct: 7.5,
};

export const spatialCrossValidation = {
  folds: [
    { label: 'Right holdout', iou: 23.9 },
    { label: 'Left holdout', iou: 27.4 },
    { label: 'Top holdout', iou: 40.8 },
    { label: 'Bottom holdout', iou: 26.2 },
  ],
  meanIou: 29.6,
  stdIou: 6.6,
  note: '4 epochs/fold (time-limited) — read the spread, not the absolute values. Tight enough to show the architecture generalizes across regions rather than overfitting one split.',
};

// Domain robustness reuses dashboard/src/data/domain_shift_results.json for
// the numbers, this just adds the demo-facing interpretation text.
export const domainRobustnessInterpretation =
  'The model is robust to moderate lighting and hue changes but highly sensitive to severe haze/fog and combined worst-case shifts, where IoU collapses by over 90%. This motivates the Image Quality check surfaced in the System Status layer — poor-quality imagery should trigger a warning rather than silently degrading detections.';

export const inferencePerformance = {
  perPatchLatencyMs: { noTta: 58, withTta: 227 },
  fullTileSeconds: { noTta: 6.8, withTta: 27.0 },
  throughputHaPerHour: { noTta: 13328, withTta: 3332 },
  typicalVillageSeconds: { noTta: 27, withTta: 108 },
  environment: 'Single CPU core, no GPU required',
  imageDimensions: '1000 × 1000 px demo tile (0.5 m/px)',
};

export const changeDetection = {
  newConstructionPainted: 2,
  newConstructionFlagged: 1,
  demolitionSimulated: 1,
  demolitionFlagged: 1,
  note: 'Missed one small, oddly-colored structure in forest cover — an explainable failure mode, not a silent one.',
};
