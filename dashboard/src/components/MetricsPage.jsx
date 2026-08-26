import React, { useState } from 'react';
import { XIcon, LayersIcon } from './Icons';
import domainShiftData from '../data/domain_shift_results.json';
import {
  buildingSegmentation,
  improvementJourney,
  ensembleComparison,
  spatialCrossValidation,
  domainRobustnessInterpretation,
  inferencePerformance,
} from '../data/metrics';
import { computeReviewStats } from '../utils/systemStatus';

function MetricTile({ label, value, color = '#c9d6e3' }) {
  return (
    <div className="border border-gcs-border bg-slate-950/50 px-3 py-2 flex flex-col gap-1">
      <span className="font-mono text-[10px] text-gcs-dim tracking-wider">{label}</span>
      <span className="font-mono text-lg font-bold" style={{ color }}>
        {value}
      </span>
    </div>
  );
}

function IouBar({ label, iou, max = 45, highlight }) {
  const pct = Math.max(0, Math.min(100, (iou / max) * 100));
  return (
    <div className="mb-2.5">
      <div className="flex justify-between mb-1">
        <span className={`font-mono text-[11px] ${highlight ? 'text-gcs-cyan font-bold' : 'text-gcs-text'}`}>
          {label}
        </span>
        <span className={`font-mono text-[11px] font-bold ${highlight ? 'text-gcs-cyan' : 'text-white'}`}>
          {iou.toFixed(1)}%
        </span>
      </div>
      <div className="domain-shift-bar">
        <div
          className="domain-shift-fill"
          style={{ width: `${pct}%`, backgroundColor: highlight ? '#00d4ff' : '#4a6075' }}
        />
      </div>
    </div>
  );
}

function SectionHeader({ title, subtitle }) {
  return (
    <div className="mb-3">
      <div className="font-mono text-xs font-bold tracking-widest text-gcs-cyan">{title}</div>
      {subtitle && <div className="font-mono text-[10px] text-gcs-dim mt-0.5">{subtitle}</div>}
    </div>
  );
}

const TABS = [
  { id: 'segmentation', label: 'SEGMENTATION' },
  { id: 'ensemble', label: 'ENSEMBLE' },
  { id: 'spatial', label: 'SPATIAL CV' },
  { id: 'domain', label: 'DOMAIN ROBUSTNESS' },
  { id: 'inference', label: 'INFERENCE' },
  { id: 'hitl', label: 'HUMAN-IN-THE-LOOP' },
];

export default function MetricsPage({ buildings, onClose }) {
  const [tab, setTab] = useState('segmentation');
  const review = computeReviewStats(buildings);
  const baselineIou = domainShiftData[0]?.iou || 1;

  return (
    <div className="ml-modal" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="ml-modal-panel">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gcs-border bg-slate-950/80 shrink-0">
          <div className="flex items-center gap-2">
            <LayersIcon className="w-4 h-4 text-gcs-purple" />
            <span className="font-mono text-xs font-bold tracking-widest text-gcs-purple">
              OVERALL METRICS
            </span>
            <span className="font-mono text-[10px] text-gcs-dim ml-1">
              Model Performance → Ensemble → Spatial CV → Domain Robustness → Inference → Human-in-the-Loop
            </span>
          </div>
          <button
            onClick={onClose}
            className="font-mono text-xs text-gcs-dim hover:text-gcs-crimson transition-colors p-1"
            title="Close Metrics"
          >
            <XIcon className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 px-4 pt-2.5 pb-0 shrink-0 border-b border-gcs-border overflow-x-auto">
          {TABS.map(t => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`ml-tab-btn ${tab === t.id ? 'active' : ''}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto min-h-0 p-4">
          {/* ── Building Segmentation ─────────────────────────────────── */}
          {tab === 'segmentation' && (
            <div>
              <SectionHeader
                title="BUILDING SEGMENTATION"
                subtitle="Final 3-model ensemble, measured on a spatially disjoint held-out region (zero data leakage)."
              />
              <div className="grid grid-cols-4 gap-2 mb-5">
                <MetricTile label="IoU" value={`${buildingSegmentation.iou}%`} color="#00d4ff" />
                <MetricTile label="Precision" value={`${buildingSegmentation.precision}%`} color="#00ff88" />
                <MetricTile label="Recall" value={`${buildingSegmentation.recall}%`} color="#ffaa00" />
                <MetricTile label="F1 Score" value={`${buildingSegmentation.f1}%`} color="#a855f7" />
              </div>

              <div className="font-mono text-[11px] text-gcs-dim mb-2 tracking-wider">
                IMPROVEMENT JOURNEY
              </div>
              {improvementJourney.map(stage => (
                <div key={stage.stage} className="mb-2.5">
                  <IouBar label={stage.stage} iou={stage.iou} highlight={stage.stage.startsWith('Final')} />
                  <div className="font-mono text-[10px] text-gcs-dim -mt-1.5">{stage.note}</div>
                </div>
              ))}
            </div>
          )}

          {/* ── Ensemble Comparison ───────────────────────────────────── */}
          {tab === 'ensemble' && (
            <div>
              <SectionHeader
                title="ENSEMBLE vs. INDIVIDUAL MODELS"
                subtitle="Averaging 3 independently-seeded models beats every individual member."
              />
              {ensembleComparison.members.map(m => (
                <IouBar key={m.label} label={m.label} iou={m.iou} />
              ))}
              <IouBar
                label={ensembleComparison.ensemble.label}
                iou={ensembleComparison.ensemble.iou}
                highlight
              />
              <div className="mt-4 border border-gcs-cyan/30 bg-gcs-cyan/5 px-3 py-2 font-mono text-xs text-gcs-cyan">
                +{ensembleComparison.relativeImprovementPct}% IoU improvement from ensembling, relative to the
                best single model.
              </div>
            </div>
          )}

          {/* ── Spatial Cross-Validation ──────────────────────────────── */}
          {tab === 'spatial' && (
            <div>
              <SectionHeader
                title="SPATIAL CROSS-VALIDATION"
                subtitle="4-fold spatial CV — held-out region rotated through all 4 tile edges (4 epochs/fold, time-limited)."
              />
              {spatialCrossValidation.folds.map(f => (
                <IouBar key={f.label} label={f.label} iou={f.iou} />
              ))}
              <div className="grid grid-cols-2 gap-2 mt-4">
                <MetricTile label="MEAN IoU" value={`${spatialCrossValidation.meanIou}%`} color="#00d4ff" />
                <MetricTile label="STD DEV" value={`± ${spatialCrossValidation.stdIou}%`} color="#4a6075" />
              </div>
              <div className="font-mono text-[10px] text-gcs-dim mt-3 leading-relaxed">
                {spatialCrossValidation.note}
              </div>
            </div>
          )}

          {/* ── Domain Robustness ─────────────────────────────────────── */}
          {tab === 'domain' && (
            <div>
              <SectionHeader
                title="DOMAIN ROBUSTNESS — STRESS TEST"
                subtitle="Same 3-model ensemble, re-run on synthetically perturbed imagery."
              />
              {domainShiftData.map(entry => {
                const iouPct = entry.iou * 100;
                const isBaseline = entry.delta_pct === 0;
                return (
                  <IouBar
                    key={entry.condition}
                    label={entry.condition}
                    iou={iouPct}
                    max={Math.max(baselineIou * 100, 45)}
                    highlight={isBaseline}
                  />
                );
              })}
              <div className="mt-4 border border-gcs-amber/30 bg-gcs-amber/5 px-3 py-2 font-mono text-[11px] text-gcs-amber leading-relaxed">
                {domainRobustnessInterpretation}
              </div>
            </div>
          )}

          {/* ── Inference Performance ─────────────────────────────────── */}
          {tab === 'inference' && (
            <div>
              <SectionHeader
                title="INFERENCE PERFORMANCE"
                subtitle={`${inferencePerformance.environment} · ${inferencePerformance.imageDimensions}`}
              />
              <div className="grid grid-cols-2 gap-3">
                <div className="border border-gcs-border p-3">
                  <div className="font-mono text-[10px] text-gcs-dim mb-2 tracking-wider">NO TTA</div>
                  <Row2 label="Per-patch latency" value={`${inferencePerformance.perPatchLatencyMs.noTta} ms`} />
                  <Row2 label="Full tile (1000×1000px)" value={`${inferencePerformance.fullTileSeconds.noTta} s`} />
                  <Row2 label="Throughput" value={`${inferencePerformance.throughputHaPerHour.noTta.toLocaleString()} ha/hr`} />
                  <Row2 label="Typical village (~100ha)" value={`~${inferencePerformance.typicalVillageSeconds.noTta}s`} />
                </div>
                <div className="border border-gcs-cyan/40 p-3 bg-gcs-cyan/5">
                  <div className="font-mono text-[10px] text-gcs-cyan mb-2 tracking-wider">WITH TTA (4-way)</div>
                  <Row2 label="Per-patch latency" value={`${inferencePerformance.perPatchLatencyMs.withTta} ms`} accent />
                  <Row2 label="Full tile (1000×1000px)" value={`${inferencePerformance.fullTileSeconds.withTta} s`} accent />
                  <Row2 label="Throughput" value={`${inferencePerformance.throughputHaPerHour.withTta.toLocaleString()} ha/hr`} accent />
                  <Row2 label="Typical village (~100ha)" value={`~${inferencePerformance.typicalVillageSeconds.withTta}s`} accent />
                </div>
              </div>
            </div>
          )}

          {/* ── Human-in-the-Loop ─────────────────────────────────────── */}
          {tab === 'hitl' && (
            <div>
              <SectionHeader
                title="HUMAN-IN-THE-LOOP PERFORMANCE"
                subtitle="Computed live from the buildings currently loaded in the app."
              />
              <div className="grid grid-cols-3 gap-2 mb-3">
                <MetricTile label="Total Detections" value={review.total} />
                <MetricTile label="Auto Processable" value={review.autoProcessable} color="#00ff88" />
                <MetricTile label="Review Required" value={review.reviewRequired} color="#ffaa00" />
              </div>
              <div className="grid grid-cols-3 gap-2 mb-3">
                <MetricTile label="High Priority" value={review.high} color="#ff3355" />
                <MetricTile label="Medium Priority" value={review.medium} color="#ffaa00" />
                <MetricTile label="Review Rate" value={`${review.reviewRate.toFixed(1)}%`} />
              </div>
              <div className="grid grid-cols-2 gap-2">
                <MetricTile label="Accepted (Human)" value={review.accepted} color="#00ff88" />
                <MetricTile label="Rejected (Human)" value={review.rejected} color="#ff3355" />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function Row2({ label, value, accent }) {
  return (
    <div className="flex justify-between mb-1.5">
      <span className="font-mono text-[10px] text-gcs-dim">{label}</span>
      <span className={`font-mono text-[11px] font-bold ${accent ? 'text-gcs-cyan' : 'text-white'}`}>
        {value}
      </span>
    </div>
  );
}
