import React from 'react';
import { ShieldAlertIcon, ImageIcon } from './Icons';

export default function LayerControls({
  layerVis,
  onToggleLayer,
  thematicMode,
  onChangeThematicMode,
  buildingCounts,
  roadCount,
  waterCount,
  triageMode,
  onToggleTriageMode,
  onOpenGallery,
}) {
  return (
    <div
      id="left-panel"
      className="gcs-panel absolute left-4 top-28 z-[1000] w-[210px] p-0 select-none pointer-events-auto"
    >
      {/* Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gcs-border">
        <svg width="10" height="10" viewBox="0 0 10 10">
          <rect width="10" height="10" fill="#00d4ff" opacity="0.8" />
        </svg>
        <span className="font-mono text-xs font-bold tracking-widest text-gcs-cyan">
          VECTOR LAYERS
        </span>
      </div>

      {/* Layer Toggles */}
      <div className="px-3 py-2 space-y-2">
        {/* Buildings */}
        <label className="toggle-wrap flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={layerVis.buildings}
            onChange={e => onToggleLayer('buildings', e.target.checked)}
          />
          <div className="toggle-track">
            <div className="toggle-thumb" />
          </div>
          <span className="w-2 h-2 flex-shrink-0 bg-[#ff3355]" />
          <span className="font-mono text-xs text-gcs-text">BUILDINGS</span>
          <span className="font-mono text-xs text-gcs-dim ml-auto">{buildingCounts}</span>
        </label>

        {/* Roads */}
        <label className="toggle-wrap flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={layerVis.roads}
            onChange={e => onToggleLayer('roads', e.target.checked)}
          />
          <div className="toggle-track">
            <div className="toggle-thumb" />
          </div>
          <span className="w-2 h-2 flex-shrink-0 bg-[#ffaa00]" />
          <span className="font-mono text-xs text-gcs-text">ROADS</span>
          <span className="font-mono text-xs text-gcs-dim ml-auto">{roadCount}</span>
        </label>

        {/* Water */}
        <label className={`toggle-wrap flex items-center gap-2 ${waterCount > 0 ? 'cursor-pointer' : 'cursor-not-allowed opacity-45'}`}>
          <input
            type="checkbox"
            checked={layerVis.water && waterCount > 0}
            disabled={waterCount === 0}
            onChange={e => onToggleLayer('water', e.target.checked)}
          />
          <div className="toggle-track">
            <div className="toggle-thumb" />
          </div>
          <span className="w-2 h-2 flex-shrink-0 bg-[#00d4ff]" />
          <span className="font-mono text-xs text-gcs-text">WATER</span>
          <span className="font-mono text-xs text-gcs-dim ml-auto">{waterCount === 0 ? 'NO EVIDENCE' : waterCount}</span>
        </label>
      </div>

      {/* Thematic Overlays */}
      <div className="border-t border-gcs-border px-3 py-2">
        <div className="font-mono text-xs text-gcs-dim mb-2 tracking-wider">
          THEMATIC OVERLAY
        </div>
        <div className="space-y-1">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="overlay"
              value="mat"
              checked={thematicMode === 'mat'}
              onChange={() => onChangeThematicMode('mat')}
              className="accent-gcs-cyan w-3 h-3"
            />
            <span className="font-mono text-xs text-gcs-text">MAT (Roof Material)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="overlay"
              value="tax"
              checked={thematicMode === 'tax'}
              onChange={() => onChangeThematicMode('tax')}
              className="accent-gcs-cyan w-3 h-3"
            />
            <span className="font-mono text-xs text-gcs-text">TAX (Heatmap)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="overlay"
              value="solar"
              checked={thematicMode === 'solar'}
              onChange={() => onChangeThematicMode('solar')}
              className="accent-gcs-cyan w-3 h-3"
            />
            <span className="font-mono text-xs text-gcs-text">SOLAR (kWp)</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="radio"
              name="overlay"
              value="conf"
              checked={thematicMode === 'conf'}
              onChange={() => onChangeThematicMode('conf')}
              className="accent-gcs-cyan w-3 h-3"
            />
            <span className="font-mono text-xs text-gcs-text">CONF (AI Score)</span>
          </label>
        </div>
      </div>

      {/* Dynamic Legend */}
      <div className="border-t border-gcs-border px-3 py-2">
        <div className="font-mono text-xs text-gcs-dim mb-2 tracking-wider">
          LEGEND ({thematicMode.toUpperCase()})
        </div>
        {thematicMode === 'mat' && (
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#ff3355] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">RCC (Concrete)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#ffaa00] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Tiled</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00d4ff] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Tin/Metal Sheet</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#a855f7] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Other</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00ff88] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Accepted</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00d4ff] flex-shrink-0 opacity-50 ring-1 ring-gcs-cyan" />
              <span className="font-mono text-xs text-gcs-dim">Rect-snapped</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#a855f7] flex-shrink-0 opacity-50 ring-1 ring-gcs-purple" />
              <span className="font-mono text-xs text-gcs-dim">Edge-straight</span>
            </div>
          </div>
        )}

        {thematicMode === 'tax' && (
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#ff3355] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">High Tax Slabs</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#ffaa00] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Mid Tax Slabs</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00d4ff] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Low Tax Slabs</span>
            </div>
          </div>
        )}

        {thematicMode === 'solar' && (
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00ff88] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">High Potential (&gt;30kWp)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00d4ff] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Moderate Potential</span>
            </div>
          </div>
        )}

        {thematicMode === 'conf' && (
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#00ff88] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">High Conf (&ge;80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#ffaa00] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Mid Conf (60-80%)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 bg-[#ff3355] flex-shrink-0" />
              <span className="font-mono text-xs text-gcs-dim">Low Conf (&lt;60%)</span>
            </div>
          </div>
        )}
      </div>

      {/* Tax rate reference */}
      <div className="border-t border-gcs-border px-3 py-2">
        <div className="font-mono text-xs text-gcs-dim mb-1 tracking-wider">
          TAX SLAB (INR/m²/yr)
        </div>
        <div className="font-mono text-xs text-gcs-dim leading-5">
          RCC&nbsp;&nbsp;₹45 &nbsp;·&nbsp; Tile ₹28<br />
          Tin&nbsp;&nbsp;₹18 &nbsp;·&nbsp; Other ₹15
        </div>
      </div>

      {/* Triage Mode Toggle */}
      <div className="border-t border-gcs-border px-3 py-2">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            id="triage-mode-toggle"
            checked={!!triageMode}
            onChange={e => onToggleTriageMode && onToggleTriageMode(e.target.checked)}
          />
          <div className="toggle-track">
            <div className="toggle-thumb" />
          </div>
          <ShieldAlertIcon
            className={`w-3.5 h-3.5 flex-shrink-0 ${
              triageMode ? 'text-gcs-amber' : 'text-gcs-dim'
            }`}
          />
          <span
            className={`font-mono text-xs ${
              triageMode ? 'text-gcs-amber font-bold' : 'text-gcs-text'
            }`}
          >
            TRIAGE MODE
          </span>
        </label>
        {triageMode && (
          <div className="font-mono text-[10px] text-gcs-amber mt-1 ml-6 leading-4">
            Showing flagged buildings only · non-flagged dimmed
          </div>
        )}
      </div>

      {/* ML Gallery Button */}
      <div className="border-t border-gcs-border px-3 py-2">
        <button
          id="open-ml-gallery-btn"
          onClick={onOpenGallery}
          className="w-full flex items-center gap-2 px-2 py-1.5 border border-gcs-border hover:border-gcs-cyan bg-slate-900/60 hover:bg-gcs-cyan/10 transition-all rounded font-mono text-xs text-gcs-dim hover:text-gcs-cyan"
        >
          <ImageIcon className="w-3.5 h-3.5 flex-shrink-0" />
          ML GALLERY
          <span className="ml-auto text-[9px] text-gcs-dim">v4.0</span>
        </button>
      </div>
    </div>
  );
}
