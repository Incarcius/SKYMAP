import React from 'react';
import { XIcon, ImageIcon, LayersIcon } from './Icons';
import { VILLAGE_OPTIONS } from '../data/villageOptions';

/**
 * Full-screen (or modal, via `overlay`) picker for choosing which demo
 * region/photo to load. Selecting a card re-runs the load pipeline against
 * a real pixel-filtered subset of the dataset (see utils/regionFilter.js) —
 * this is what demonstrates the pipeline's flexibility across different
 * building densities/layouts, not just a single fixed screenshot.
 */
export default function VillagePicker({ onSelect, onClose, activeId, dismissible }) {
  return (
    <div
      className="ml-modal"
      onClick={e => {
        if (dismissible && e.target === e.currentTarget) onClose?.();
      }}
    >
      <div className="ml-modal-panel survey-picker-panel" style={{ maxWidth: 760 }}>
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-2.5 border-b border-gcs-border bg-slate-950/80 shrink-0">
          <div className="flex items-center gap-2">
            <ImageIcon className="w-4 h-4 text-gcs-cyan" />
            <span className="font-mono text-xs font-bold tracking-widest text-gcs-cyan">
              SELECT AERIAL SURVEY AREA
            </span>
            <span className="font-mono text-[10px] text-gcs-dim ml-1">
              Independent survey inputs · select an area to analyze
            </span>
          </div>
          {dismissible && (
            <button
              onClick={onClose}
              className="font-mono text-xs text-gcs-dim hover:text-gcs-crimson transition-colors p-1"
              title="Close"
            >
              <XIcon className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Body: 2x2 photo grid */}
        <div className="flex-1 overflow-y-auto p-4">
          <div className="grid grid-cols-2 gap-3">
            {VILLAGE_OPTIONS.map(opt => {
              const isActive = opt.id === activeId;
              return (
                <button
                  key={opt.id}
                  onClick={() => onSelect(opt.id)}
                  className={`text-left border transition-all overflow-hidden rounded flex flex-col ${
                    isActive
                      ? 'border-gcs-cyan bg-gcs-cyan/5'
                      : 'border-gcs-border bg-slate-900/60 hover:border-slate-500'
                  }`}
                >
                  <div className="relative w-full aspect-[16/9] bg-black overflow-hidden flex-shrink-0">
                    <img
                      src={opt.thumbnail}
                      alt={opt.name}
                      className="w-full h-full object-cover"
                      draggable={false}
                    />
                    {isActive && (
                      <div className="absolute top-1.5 right-1.5 flex items-center gap-1 px-1.5 py-0.5 bg-gcs-cyan text-black font-mono text-[9px] font-bold rounded">
                        <LayersIcon className="w-2.5 h-2.5" />
                        LOADED
                      </div>
                    )}
                  </div>
                  <div className="p-2.5 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="font-mono text-xs font-bold text-white leading-tight">{opt.name}</span>
                      <span className={`font-mono text-[9px] px-1.5 py-0.5 rounded border ${opt.quality === 'GOOD' ? 'text-gcs-green border-gcs-green/30 bg-gcs-green/5' : 'text-gcs-amber border-gcs-amber/30 bg-gcs-amber/5'}`}>{opt.quality} · {opt.qualityScore}</span>
                    </div>
                    <div className="font-mono text-[10px] text-gcs-cyan">{opt.subtitle}</div>
                    <div className="font-mono text-[10px] text-gcs-dim leading-snug">{opt.description}</div>
                    <div className="flex items-center gap-2 pt-1 font-mono text-[9px] text-slate-500"><span>BUILDINGS</span><span>ROADS</span><span>{opt.featureAvailability?.water ? 'WATER' : 'NO WATER'}</span></div>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
