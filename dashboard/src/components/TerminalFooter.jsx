import React from 'react';

export default function TerminalFooter({ cursorCoords }) {
  const latStr = cursorCoords ? cursorCoords.lat.toFixed(4) : '--.----';
  const lngStr = cursorCoords ? cursorCoords.lng.toFixed(4) : '--.----';

  return (
    <footer className="h-7 w-full bg-slate-950 border-t border-slate-800 flex items-center justify-between px-4 text-[10px] font-mono text-slate-500 shrink-0 z-[1000] select-none">
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

      {/* Footer Right: System Uplink Status */}
      <div className="flex items-center gap-2">
        <span className="text-slate-400">
          SYS: <span className="text-gcs-green">ACTIVE</span> &nbsp;|&nbsp; UPLINK:{' '}
          <span className="text-gcs-cyan">STABLE</span>
        </span>
      </div>
    </footer>
  );
}
