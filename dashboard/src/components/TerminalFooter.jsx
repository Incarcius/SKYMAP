import React from 'react';
import { LayersIcon, ImageIcon, ShieldAlertIcon } from './Icons';

export default function TerminalFooter({ cursorCoords, onOpenGallery, onOpenMetrics, onExportCsv, buildings, triageMode }) {
  const latStr = cursorCoords ? cursorCoords.lat.toFixed(4) : '--.----';
  const lngStr = cursorCoords ? cursorCoords.lng.toFixed(4) : '--.----';

  return (
    <footer className="h-10 w-full bg-slate-950 border-t border-slate-800 flex items-center justify-between px-4 text-xs font-mono text-slate-500 shrink-0 z-[1000] select-none">
      {/* Footer Left: Live Cursor Coordinates */}
      <div className="flex items-center gap-2">
        <span className="w-1.5 h-1.5 rounded-full bg-gcs-cyan animate-pulse" />
        <span className="text-slate-400">
          LAT: <span className="text-gcs-cyan">{latStr}</span> &nbsp;|&nbsp; LON:{' '}
          <span className="text-gcs-cyan">{lngStr}</span>
        </span>
      </div>

      {/* Footer Center: Throughput Benchmark */}
      <div className="flex items-center gap-2">
        <span className="text-slate-400 font-semibold tracking-wider">
          THROUGHPUT: <span className="text-gcs-green font-bold">13,328 ha/hr</span>
        </span>
      </div>

      {/* Footer Right: System Uplink Status & Actions */}
      <div className="flex items-center h-full">
        {/* Triage Active Indicator */}
        {triageMode && (
          <div className="h-full flex items-center gap-1.5 px-4 border-l border-slate-800 shrink-0">
            <ShieldAlertIcon className="w-3.5 h-3.5 text-gcs-amber" />
            <span className="blink font-mono text-[11px] font-bold tracking-widest text-gcs-amber mt-0.5">TRIAGE ACTIVE</span>
          </div>
        )}

        {/* CSV Export */}
        <button
          onClick={onExportCsv}
          disabled={!buildings || !buildings.length}
          className="h-full flex items-center gap-2 px-4 hover:text-gcs-green hover:bg-slate-900 border-l border-slate-800 disabled:opacity-30 disabled:cursor-not-allowed transition-colors font-bold tracking-widest text-[11px]"
          title="Export current survey results as CSV"
        >
          <span className="download-glyph w-4 h-4 text-sm" aria-hidden="true">↓</span> CSV
        </button>

        {/* Overall Metrics Button */}
        <button
          onClick={onOpenMetrics}
          className="h-full flex items-center gap-2 px-4 hover:text-gcs-purple hover:bg-slate-900 border-l border-slate-800 transition-colors font-bold tracking-widest text-[11px]"
          title="Open Overall Metrics"
        >
          <LayersIcon className="w-4 h-4" /> METRICS
        </button>

        {/* ML Gallery Button */}
        <button
          onClick={onOpenGallery}
          className="h-full flex items-center gap-2 px-4 hover:text-gcs-cyan hover:bg-slate-900 border-l border-slate-800 transition-colors font-bold tracking-widest text-[11px]"
          title="Open ML Gallery"
        >
          <ImageIcon className="w-4 h-4" /> GALLERY
        </button>

        <div className="h-full flex items-center border-l border-slate-800 pl-4">
          <span className="text-slate-400">
            SYS: <span className="text-gcs-green">ACTIVE</span> &nbsp;|&nbsp; UPLINK:{' '}
            <span className="text-gcs-cyan">STABLE</span>
          </span>
        </div>
      </div>
    </footer>
  );
}
