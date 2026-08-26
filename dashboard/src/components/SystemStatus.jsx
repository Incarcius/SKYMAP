import React from 'react';
import { ShieldAlertIcon, LayersIcon } from './Icons';
import { APP_STATUS } from '../utils/systemStatus';

const STATUS_COLORS = {
  [APP_STATUS.OPERATIONAL]: '#00ff88',
  [APP_STATUS.PROCESSING]: '#00d4ff',
  [APP_STATUS.WARNING]: '#ffaa00',
  [APP_STATUS.ERROR]: '#ff3355',
};

function Row({ label, value, color }) {
  return (
    <div className="flex justify-between items-baseline">
      <span className="font-mono text-[10px] text-gcs-dim">{label}</span>
      <span className="font-mono text-xs font-bold" style={{ color: color || '#c9d6e3' }}>
        {value}
      </span>
    </div>
  );
}

/**
 * Purely presentational. All numbers come from `status`, produced by
 * utils/systemStatus.js:computeSystemStatus(). This component never
 * recomputes counts or averages itself.
 */
export default function SystemStatus({ status, collapsed, onToggleCollapsed }) {
  if (!status) return null;

  const color = STATUS_COLORS[status.status] || '#4a6075';

  return (
    <div
      id="system-status-panel"
      className="gcs-panel absolute left-4 bottom-14 z-[1000] w-[220px] p-0 select-none pointer-events-auto"
    >
      {/* Header */}
      <button
        onClick={onToggleCollapsed}
        className="w-full flex items-center gap-2 px-3 py-2 border-b border-gcs-border justify-between bg-slate-950/60"
      >
        <div className="flex items-center gap-2">
          <LayersIcon className="w-3.5 h-3.5" style={{ color }} />
          <span className="font-mono text-xs font-bold tracking-widest" style={{ color }}>
            SYSTEM STATUS
          </span>
        </div>
        <span
          className={`w-2 h-2 rounded-full ${status.status === APP_STATUS.PROCESSING ? 'blink' : ''}`}
          style={{ backgroundColor: color }}
        />
      </button>

      {!collapsed && (
        <div className="px-3 py-2 space-y-1.5">
          <Row label="STATUS" value={status.status} color={color} />
          <Row label="APPLICATION" value={status.application} />
          <Row label="MODEL" value={status.model} />
          <Row label="MODE" value={status.mode} />
          <Row label="DATASET" value={status.dataset} />

          {status.isProcessing ? (
            <div className="border-t border-gcs-border pt-2 mt-1 space-y-1">
              {status.stages.map((stage, i) => {
                const done = i < status.currentStageIndex;
                const active = i === status.currentStageIndex;
                const stColor = done ? '#00ff88' : active ? '#00d4ff' : '#4a6075';
                return (
                  <div key={stage.key} className="flex items-center gap-2">
                    <span
                      className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${active ? 'blink' : ''}`}
                      style={{ backgroundColor: stColor }}
                    />
                    <span className="font-mono text-[10px]" style={{ color: stColor }}>
                      {stage.label}
                      {done ? ' ✓' : active ? '…' : ''}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : (
            <>
              <div className="border-t border-gcs-border my-1.5" />
              <Row label="BUILDINGS" value={status.buildings} />
              <Row label="ROADS" value={status.roads} />
              <Row label="WATER BODIES" value={status.waterBodies === 0 ? '0' : status.waterBodies} />

              <div className="border-t border-gcs-border my-1.5" />
              <Row label="HIGH PRIORITY" value={status.highPriority} color={status.highPriority > 0 ? '#ff3355' : undefined} />
              <Row label="MEDIUM PRIORITY" value={status.mediumPriority} color={status.mediumPriority > 0 ? '#ffaa00' : undefined} />
              <Row label="AUTO PROCESSABLE" value={status.autoProcessable} color="#00ff88" />

              <div className="border-t border-gcs-border my-1.5" />
              <Row label="AVG CONFIDENCE" value={`${status.averageConfidencePct.toFixed(1)}%`} />
              <Row
                label="IMAGE QUALITY"
                value={status.imageQuality}
                color={status.imageQuality === 'GOOD' ? '#00ff88' : status.imageQuality === 'FAIR' ? '#ffaa00' : '#ff3355'}
              />
              <Row label="PROCESSING TIME" value={`${status.processingTimeSec.toFixed(1)}s`} />

              {status.status === APP_STATUS.WARNING && (
                <div className="flex items-start gap-1.5 mt-2 pt-2 border-t border-gcs-border">
                  <ShieldAlertIcon className="w-3.5 h-3.5 text-gcs-amber flex-shrink-0 mt-0.5" />
                  <span className="font-mono text-[10px] text-gcs-amber leading-tight">
                    Review load or imagery quality needs attention.
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
