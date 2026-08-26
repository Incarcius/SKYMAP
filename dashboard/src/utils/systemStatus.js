// src/utils/systemStatus.js
//
// Pure calculation layer for the System Status feature.
// SystemStatus.jsx must ONLY render what this file computes — it must not
// compute counts/averages itself. This keeps "data" and "display" separate,
// per the spec.

export const APP_STATUS = {
  PROCESSING: 'PROCESSING',
  OPERATIONAL: 'OPERATIONAL',
  WARNING: 'WARNING',
  ERROR: 'ERROR',
};

// Ordered pipeline stages shown while status === PROCESSING.
// App.jsx drives `currentStageIndex` on mount (or on "load village") using a timer;
// this file just describes what each stage means so the UI has a single source of truth.
export const PROCESSING_STAGES = [
  { key: 'init', label: 'Initializing System' },
  { key: 'imagery', label: 'Processing Imagery' },
  { key: 'buildings', label: 'Extracting Buildings' },
  { key: 'roads_water', label: 'Extracting Roads & Water Bodies' },
  { key: 'confidence', label: 'Running Confidence Analysis' },
  { key: 'review', label: 'Computing Review Priorities' },
];

/**
 * Human-in-the-loop stats, shared by SystemStatus.jsx and MetricsPage.jsx
 * so the two views can never disagree with each other.
 */
export function computeReviewStats(buildings) {
  const total = buildings.length;
  const high = buildings.filter(b => b.review_priority?.startsWith('HIGH')).length;
  const medium = buildings.filter(b => b.review_priority?.startsWith('MEDIUM')).length;
  const low = total - high - medium;

  const accepted = buildings.filter(b => b.status === 'accepted').length;
  const rejected = buildings.filter(b => b.status === 'rejected').length;

  // "Auto processable" = anything not flagged (LOW priority) that hasn't been
  // manually rejected.
  const autoProcessable = buildings.filter(
    b => (b.review_priority?.startsWith('LOW') ?? true) && b.status !== 'rejected'
  ).length;

  const reviewRequired = high + medium;
  const reviewRate = total > 0 ? (reviewRequired / total) * 100 : 0;

  const avgConfidence =
    total > 0
      ? buildings.reduce((s, b) => s + (b.mean_pixel_probability || 0), 0) / total
      : 0;

  return {
    total,
    high,
    medium,
    low,
    accepted,
    rejected,
    autoProcessable,
    reviewRequired,
    reviewRate,
    avgConfidence,
  };
}

/**
 * Heuristic image-quality label, derived from the same confidence distribution
 * the model already produced (no separate image-quality model exists, so this
 * is explicitly a proxy — kept honest rather than inventing a fake sensor).
 */
function computeImageQuality(avgConfidence, reviewRatePct) {
  if (avgConfidence >= 0.85 && reviewRatePct <= 10) return 'GOOD';
  if (avgConfidence >= 0.7 && reviewRatePct <= 25) return 'FAIR';
  return 'POOR';
}

/**
 * Builds the single System Status object the UI renders.
 *
 * @param {Array} buildings
 * @param {Array} roads
 * @param {Array} water
 * @param {Object} opts
 * @param {string} opts.appStatus        one of APP_STATUS
 * @param {number} opts.currentStageIndex index into PROCESSING_STAGES (while PROCESSING)
 * @param {string} opts.datasetName
 * @param {string} opts.mode             'DEMO' | 'LIVE'
 * @param {number} opts.processingTimeSec actual/simulated wall-clock time for this run
 */
export function computeSystemStatus(buildings, roads, water, opts = {}) {
  const {
    appStatus = APP_STATUS.OPERATIONAL,
    currentStageIndex = PROCESSING_STAGES.length - 1,
    datasetName = 'Demo Village',
    mode = 'DEMO',
    processingTimeSec = 6.8,
  } = opts;

  const review = computeReviewStats(buildings);
  const imageQuality = computeImageQuality(review.avgConfidence, review.reviewRate);

  // WARNING takes over OPERATIONAL when the pipeline finished but something
  // needs attention (e.g. unusually high review load or poor imagery).
  let resolvedStatus = appStatus;
  if (appStatus === APP_STATUS.OPERATIONAL) {
    if (imageQuality === 'POOR' || review.reviewRate > 25) {
      resolvedStatus = APP_STATUS.WARNING;
    }
  }

  return {
    status: resolvedStatus,
    application: 'SVAMITVA AI v0.9',
    model: 'Ensemble v1.0',
    mode,
    dataset: datasetName,

    isProcessing: appStatus === APP_STATUS.PROCESSING,
    stages: PROCESSING_STAGES,
    currentStageIndex,

    buildings: review.total,
    roads: roads.length,
    waterBodies: water.length,

    highPriority: review.high,
    mediumPriority: review.medium,
    autoProcessable: review.autoProcessable,
    reviewRequired: review.reviewRequired,
    reviewRatePct: review.reviewRate,

    averageConfidencePct: review.avgConfidence * 100,
    imageQuality,
    processingTimeSec,
  };
}
