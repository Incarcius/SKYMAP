import React, { useState } from 'react';
import { XIcon, ImageIcon } from './Icons';
import domainShiftData from '../data/domain_shift_results.json';

// ─── Gallery config ───────────────────────────────────────────────────────────
const GALLERY_CROPS = [
  { id: 1, label: 'CROP·01 — Dense Urban', formats: ['annotated', 'heatmap', 'toggle', 'upscaled2x'] },
  { id: 2, label: 'CROP·02 — Peri-Urban', formats: ['annotated', 'heatmap', 'toggle', 'upscaled2x'] },
  { id: 3, label: 'CROP·03 — Low-Density', formats: ['annotated', 'heatmap', 'toggle', 'upscaled2x'] },
];

const FORMAT_LABELS = {
  annotated:  'ANNOTATED',
  heatmap:    'HEATMAP',
  toggle:     'TOGGLE GIF',
  upscaled2x: 'UPSCALED 2×',
};

const DEMO_IMAGES = [
  { id: 1, zone: 'top',    label: 'ZONE·TOP' },
  { id: 2, zone: 'middle', label: 'ZONE·MID' },
  { id: 3, zone: 'bottom', label: 'ZONE·BOT' },
];

const DEMO_FORMATS = ['sidebyside', 'input', 'output'];
const DEMO_FORMAT_LABELS = { sidebyside: 'SIDE-BY-SIDE', input: 'INPUT', output: 'OUTPUT' };

const STRESS_IMAGES = [
  { file: 'stress_test_baseline.png',          label: 'BASELINE' },
  { file: 'stress_test_brighter.png',          label: 'BRIGHTER' },
  { file: 'stress_test_darker.png',            label: 'DARKER' },
  { file: 'stress_test_hazyfoggy.png',         label: 'HAZY/FOGGY' },
  { file: 'stress_test_hueshifted.png',        label: 'HUE-SHIFTED' },
  { file: 'stress_test_combined_worstcase.png',label: 'WORST-CASE' },
];

// ─── Domain Shift bar chart row ───────────────────────────────────────────────
function DomainShiftRow({ entry, baselineIou }) {
  const isBaseline = entry.delta_pct === 0;
  const iouPct = Math.round(entry.iou * 1000) / 10;
  const barPct = Math.max(0, Math.min(100, (entry.iou / baselineIou) * 100));
  const barColor =
    isBaseline      ? '#00d4ff' :
    entry.delta_pct > -30 ? '#ffaa00' :
    '#ff3355';
  const deltaColor =
    isBaseline      ? '#4a6075' :
    entry.delta_pct > -30 ? '#ffaa00' :
    '#ff3355';

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="font-mono text-[11px] text-gcs-text">{entry.condition}</span>
        <div className="flex items-center gap-3">
          <span className="font-mono text-[11px] font-bold text-white">IoU {iouPct.toFixed(1)}%</span>
          <span
            className="font-mono text-[10px] font-bold w-[72px] text-right"
            style={{ color: deltaColor }}
          >
            {isBaseline ? '(baseline)' : `${entry.delta_pct.toFixed(1)}%`}
          </span>
        </div>
      </div>
      <div className="domain-shift-bar">
        <div
          className="domain-shift-fill"
          style={{ width: `${barPct.toFixed(1)}%`, backgroundColor: barColor }}
        />
      </div>
    </div>
  );
}

// ─── Main modal ───────────────────────────────────────────────────────────────
export default function MLGalleryModal({ onClose }) {
  const [mainTab, setMainTab]           = useState('gallery');  // 'gallery' | 'domain'
  const [gallerySubTab, setGallerySubTab] = useState('crops');    // 'crops' | 'demo'
  const [activeCrop, setActiveCrop]     = useState(1);
  const [activeFormat, setActiveFormat] = useState('annotated');
  const [activeDemoId, setActiveDemoId] = useState(1);
  const [activeDemoFmt, setActiveDemoFmt] = useState('sidebyside');
  const [stressImg, setStressImg]       = useState(null);        // lightbox

  const baseline = domainShiftData[0];

  // Gallery image src
  const galleryImgSrc = `/ml_assets/demo_gallery/gallery_${activeCrop}_${activeFormat}.${activeFormat === 'toggle' ? 'gif' : 'png'}`;
  const demoZone      = DEMO_IMAGES.find(d => d.id === activeDemoId)?.zone ?? 'top';
  const demoImgSrc    = `/ml_assets/demo_images/demo_${activeDemoId}_${demoZone}_${activeDemoFmt}.png`;

  return (
    <div className="ml-modal" onClick={e => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="ml-modal-panel">

        {/* ── Modal Header ────────────────────────────────────────────────── */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gcs-border bg-slate-950/80 shrink-0">
          <div className="flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-gcs-cyan" />
            <span className="font-mono text-xs font-bold tracking-widest text-gcs-cyan">
              ML MODEL GALLERY
            </span>
            <span className="font-mono text-[10px] text-gcs-dim ml-1">v4.0 · Attention-ResUNet+ASPP · Ensemble ×3</span>
          </div>
          <button
            id="ml-gallery-close"
            onClick={onClose}
            className="font-mono text-xs text-gcs-dim hover:text-gcs-crimson transition-colors p-1"
            title="Close Gallery"
          >
            <XIcon className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* ── Main Tabs ───────────────────────────────────────────────────── */}
        <div className="flex gap-1 px-4 pt-2.5 pb-0 shrink-0 border-b border-gcs-border">
          {[
            { id: 'gallery', label: 'DEMO GALLERY' },
            { id: 'domain',  label: 'DOMAIN-SHIFT STRESS TEST' },
          ].map(t => (
            <button
              key={t.id}
              id={`ml-tab-${t.id}`}
              onClick={() => setMainTab(t.id)}
              className={`ml-tab-btn ${mainTab === t.id ? 'active' : ''}`}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* ── Body ────────────────────────────────────────────────────────── */}
        <div className="flex-1 overflow-y-auto min-h-0">

          {/* ======= DEMO GALLERY TAB ======================================= */}
          {mainTab === 'gallery' && (
            <div className="p-4 flex flex-col gap-4 h-full">
              {/* Gallery Sub-tabs */}
              <div className="flex gap-1">
                <button
                  id="gallery-subtab-crops"
                  onClick={() => setGallerySubTab('crops')}
                  className={`ml-tab-btn small ${gallerySubTab === 'crops' ? 'active' : ''}`}
                >
                  CURATED CROPS
                </button>
                <button
                  id="gallery-subtab-demo"
                  onClick={() => setGallerySubTab('demo')}
                  className={`ml-tab-btn small ${gallerySubTab === 'demo' ? 'active' : ''}`}
                >
                  SCENE COMPARISONS
                </button>
              </div>

              {/* CURATED CROPS */}
              {gallerySubTab === 'crops' && (
                <div className="flex gap-4 h-full min-h-0">
                  {/* Crop selector */}
                  <div className="flex flex-col gap-2 shrink-0 w-44">
                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider mb-1">SELECT CROP</div>
                    {GALLERY_CROPS.map(c => (
                      <button
                        key={c.id}
                        id={`crop-btn-${c.id}`}
                        onClick={() => setActiveCrop(c.id)}
                        className={`text-left px-2 py-1.5 border font-mono text-[11px] transition-all rounded ${
                          activeCrop === c.id
                            ? 'border-gcs-cyan bg-gcs-cyan/10 text-gcs-cyan'
                            : 'border-gcs-border bg-slate-900/60 text-gcs-dim hover:border-slate-500 hover:text-gcs-text'
                        }`}
                      >
                        {c.label}
                      </button>
                    ))}

                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider mt-3 mb-1">FORMAT</div>
                    {GALLERY_CROPS[0].formats.map(fmt => (
                      <button
                        key={fmt}
                        id={`format-btn-${fmt}`}
                        onClick={() => setActiveFormat(fmt)}
                        className={`text-left px-2 py-1.5 border font-mono text-[11px] transition-all rounded ${
                          activeFormat === fmt
                            ? 'border-gcs-amber bg-gcs-amber/10 text-gcs-amber'
                            : 'border-gcs-border bg-slate-900/60 text-gcs-dim hover:border-slate-500 hover:text-gcs-text'
                        }`}
                      >
                        {FORMAT_LABELS[fmt]}
                      </button>
                    ))}

                    {/* Reject samples */}
                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider mt-3 mb-1">REJECT SAMPLES</div>
                    {[1, 2].map(ri => (
                      <button
                        key={ri}
                        id={`reject-btn-${ri}`}
                        onClick={() => { setActiveCrop(-ri); }}
                        className={`text-left px-2 py-1.5 border font-mono text-[11px] transition-all rounded ${
                          activeCrop === -ri
                            ? 'border-gcs-crimson bg-gcs-crimson/10 text-gcs-crimson'
                            : 'border-gcs-border bg-slate-900/60 text-gcs-dim hover:border-slate-500 hover:text-gcs-text'
                        }`}
                      >
                        REJECT·0{ri}
                      </button>
                    ))}
                  </div>

                  {/* Image viewer */}
                  <div className="flex-1 flex flex-col gap-2 min-w-0">
                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider">
                      {activeCrop > 0
                        ? `${GALLERY_CROPS[activeCrop - 1]?.label} — ${FORMAT_LABELS[activeFormat]}`
                        : `REJECT SAMPLE·0${Math.abs(activeCrop)}`}
                    </div>
                    <div className="flex-1 bg-slate-950 border border-gcs-border rounded overflow-hidden flex items-center justify-center min-h-0">
                      {activeCrop > 0 ? (
                        <img
                          src={galleryImgSrc}
                          alt={`Gallery crop ${activeCrop} ${activeFormat}`}
                          className="max-w-full max-h-full object-contain"
                          style={{ imageRendering: activeFormat === 'upscaled2x' ? 'pixelated' : 'auto' }}
                        />
                      ) : (
                        <img
                          src={`/ml_assets/demo_gallery/reject_${Math.abs(activeCrop)}_annotated.png`}
                          alt={`Reject sample ${Math.abs(activeCrop)}`}
                          className="max-w-full max-h-full object-contain"
                        />
                      )}
                    </div>
                  </div>
                </div>
              )}

              {/* SCENE COMPARISONS */}
              {gallerySubTab === 'demo' && (
                <div className="flex gap-4 h-full min-h-0">
                  {/* Zone selector */}
                  <div className="flex flex-col gap-2 shrink-0 w-44">
                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider mb-1">SELECT ZONE</div>
                    {DEMO_IMAGES.map(d => (
                      <button
                        key={d.id}
                        id={`demo-zone-btn-${d.id}`}
                        onClick={() => setActiveDemoId(d.id)}
                        className={`text-left px-2 py-1.5 border font-mono text-[11px] transition-all rounded ${
                          activeDemoId === d.id
                            ? 'border-gcs-cyan bg-gcs-cyan/10 text-gcs-cyan'
                            : 'border-gcs-border bg-slate-900/60 text-gcs-dim hover:border-slate-500 hover:text-gcs-text'
                        }`}
                      >
                        {d.label}
                      </button>
                    ))}

                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider mt-3 mb-1">FORMAT</div>
                    {DEMO_FORMATS.map(fmt => (
                      <button
                        key={fmt}
                        id={`demo-format-btn-${fmt}`}
                        onClick={() => setActiveDemoFmt(fmt)}
                        className={`text-left px-2 py-1.5 border font-mono text-[11px] transition-all rounded ${
                          activeDemoFmt === fmt
                            ? 'border-gcs-amber bg-gcs-amber/10 text-gcs-amber'
                            : 'border-gcs-border bg-slate-900/60 text-gcs-dim hover:border-slate-500 hover:text-gcs-text'
                        }`}
                      >
                        {DEMO_FORMAT_LABELS[fmt]}
                      </button>
                    ))}
                  </div>

                  {/* Image viewer */}
                  <div className="flex-1 flex flex-col gap-2 min-w-0">
                    <div className="font-mono text-[10px] text-gcs-dim tracking-wider">
                      {DEMO_IMAGES.find(d => d.id === activeDemoId)?.label} — {DEMO_FORMAT_LABELS[activeDemoFmt]}
                    </div>
                    <div className="flex-1 bg-slate-950 border border-gcs-border rounded overflow-hidden flex items-center justify-center min-h-0">
                      <img
                        src={demoImgSrc}
                        alt={`Demo ${activeDemoId} ${activeDemoFmt}`}
                        className="max-w-full max-h-full object-contain"
                      />
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ======= DOMAIN SHIFT STRESS TEST TAB ========================== */}
          {mainTab === 'domain' && (
            <div className="p-4 flex flex-col gap-5">
              {/* Header explanation */}
              <div className="gcs-panel p-3">
                <div className="font-mono text-[11px] text-gcs-dim leading-5">
                  Synthetic transformations applied to the held-out region.
                  Each variant re-runs the full 3-model ensemble. IoU bar = fraction of baseline performance retained.
                  <span className="text-gcs-amber"> Baseline IoU: {(baseline.iou * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* IoU Delta bar chart */}
              <div>
                <div className="font-mono text-xs text-gcs-dim mb-3 tracking-wider">IoU DEGRADATION BY CONDITION</div>
                {domainShiftData.map((entry, i) => (
                  <DomainShiftRow key={i} entry={entry} baselineIou={baseline.iou} />
                ))}
              </div>

              {/* Stress test image thumbnails */}
              <div>
                <div className="font-mono text-xs text-gcs-dim mb-3 tracking-wider">VISUAL OUTPUTS (click to enlarge)</div>
                <div className="grid grid-cols-3 gap-2">
                  {STRESS_IMAGES.map((si, i) => (
                    <div
                      key={i}
                      id={`stress-thumb-${i}`}
                      className="cursor-pointer border border-gcs-border hover:border-gcs-cyan transition-colors rounded overflow-hidden"
                      onClick={() => setStressImg(si)}
                      title={si.label}
                    >
                      <img
                        src={`/ml_assets/stress/${si.file}`}
                        alt={si.label}
                        className="w-full h-20 object-cover"
                      />
                      <div className="font-mono text-[9px] text-gcs-dim text-center py-1 bg-slate-950">
                        {si.label}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Domain shift slide if it exists */}
              <div>
                <div className="font-mono text-xs text-gcs-dim mb-2 tracking-wider">DOMAIN SHIFT PRESENTATION SLIDE</div>
                <div
                  className="border border-gcs-border rounded overflow-hidden cursor-pointer hover:border-gcs-cyan transition-colors"
                  onClick={() => setStressImg({ file: 'slide_domain_shift.png', label: 'Domain Shift Slide', isSlide: true })}
                >
                  <img
                    src="/ml_assets/stress/slide_domain_shift.png"
                    alt="Domain shift slide"
                    className="w-full object-contain max-h-40"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Stress Image Lightbox ─────────────────────────────────────────── */}
      {stressImg && (
        <div
          className="fixed inset-0 z-[10000] bg-black/90 flex items-center justify-center"
          onClick={() => setStressImg(null)}
        >
          <div className="relative max-w-5xl max-h-[90vh] p-2" onClick={e => e.stopPropagation()}>
            <button
              onClick={() => setStressImg(null)}
              className="absolute top-3 right-3 z-10 p-1 bg-slate-900/80 border border-gcs-border text-gcs-dim hover:text-gcs-crimson transition-colors rounded"
            >
              <XIcon className="w-4 h-4" />
            </button>
            <div className="font-mono text-[11px] text-gcs-dim mb-2 text-center">{stressImg.label}</div>
            <img
              src={stressImg.isSlide ? '/ml_assets/stress/slide_domain_shift.png' : `/ml_assets/stress/${stressImg.file}`}
              alt={stressImg.label}
              className="max-w-full max-h-[80vh] object-contain rounded"
            />
          </div>
        </div>
      )}
    </div>
  );
}
