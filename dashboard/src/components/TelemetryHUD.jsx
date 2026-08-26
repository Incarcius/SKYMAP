import React, { useState, useEffect } from 'react';
import { LayersIcon, ImageIcon, ShieldAlertIcon } from './Icons';
import { buildingSegmentation } from '../data/metrics';

export default function TelemetryHUD({
  buildings,
  roads,
  engaged,
  onEngageToggle,
  onOpenVillagePicker,
  activeVillageName,
  triageMode,
}) {
  const [timeStr, setTimeStr] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const pad = n => String(n).padStart(2, '0');
      setTimeStr(
        `${now.getUTCFullYear()}-${pad(now.getUTCMonth() + 1)}-${pad(now.getUTCDate())} ` +
        `${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(now.getUTCSeconds())} UTC`
      );
    };
    updateTime();
    const timer = setInterval(updateTime, 1000);
    return () => clearInterval(timer);
  }, []);

  // Compute metrics dynamically from reactive buildings state
  const totalBldgs = buildings.length;
  const totalAreaM2 = buildings.reduce((acc, b) => acc + (b.area_m2 || 0), 0);
  const totalSolarKwp = buildings.reduce((acc, b) => acc + (b.estimated_solar_kwp || 0), 0);
  const totalTaxInr = buildings.reduce((acc, b) => acc + (b.estimated_annual_tax_inr || 0), 0);
  const totalRoadKm = roads.reduce((acc, r) => acc + (r.length_m || 0), 0) / 1000;

  // Auto-accepted count: explicitly accepted OR marked LOW review priority
  const autoAcceptedCount = buildings.filter(
    b => b.status === 'accepted' || (b.status !== 'rejected' && b.review_priority?.startsWith('LOW'))
  ).length;

  const autoAcceptPct = totalBldgs > 0 ? Math.round((autoAcceptedCount / totalBldgs) * 100) : 0;
  const totalAreaHa = (totalAreaM2 / 10000).toFixed(2);
  const solarMwp = (totalSolarKwp / 1000).toFixed(3);
  const taxLakh = (totalTaxInr / 100000).toFixed(1);

  return (
    <div
      className="gcs-panel shrink-0 w-full h-[44px] flex items-center px-3 gap-0 border-t-0 border-l-0 border-r-0 z-[1000] overflow-x-auto overflow-y-hidden flex-nowrap"
    >
      {/* Brand & Mission ID */}
      <div className="flex items-center gap-2 pr-3 min-w-[190px] shrink-0">
        <LayersIcon className="w-4 h-4 text-gcs-cyan" />
        <span className="font-mono text-xs font-semibold tracking-widest text-gcs-cyan">
          SKYMAP·GCS
        </span>
        <span className="font-mono text-xs text-gcs-dim">v4.0</span>
      </div>

      <div className="hud-sep" />

      {/* System Status */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">SYS</span>
        <span className={`font-mono text-xs font-bold ${engaged ? 'text-gcs-green' : 'text-gcs-amber'}`}>
          {engaged ? 'ACTIVE' : 'STANDBY'}
        </span>
        <span className={`blink w-2 h-2 rounded-full ${engaged ? 'bg-gcs-green' : 'bg-gcs-amber'}`} />
      </div>

      <div className="hud-sep" />

      {/* Mode */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">MODE</span>
        <span className="font-mono text-xs font-bold text-gcs-cyan">TRIAGE</span>
      </div>

      <div className="hud-sep" />

      {/* Active Demo Region */}
      <div className="flex items-center gap-2 px-3 shrink-0 max-w-[180px]">
        <span className="font-mono text-xs text-gcs-dim">REGION</span>
        <span className="font-mono text-xs font-bold text-white truncate" title={activeVillageName}>
          {activeVillageName}
        </span>
      </div>

      <div className="hud-sep" />

      {/* Building Count */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">BLDGS</span>
        <span className="font-mono text-xs font-bold text-white">{totalBldgs}</span>
      </div>

      <div className="hud-sep" />

      {/* Area */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">AREA</span>
        <span className="font-mono text-xs font-bold text-white">
          {totalAreaHa} <span className="font-normal text-gcs-dim">ha</span>
        </span>
      </div>

      <div className="hud-sep" />

      {/* Solar */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">SOLAR</span>
        <span className="font-mono text-xs font-bold text-gcs-green">
          {solarMwp} <span className="font-normal text-gcs-dim">MWp</span>
        </span>
      </div>

      <div className="hud-sep" />

      {/* Estimated Tax */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">EST·TAX</span>
        <span className="font-mono text-xs font-bold text-gcs-amber">
          ₹{taxLakh} <span className="font-normal text-gcs-dim">L/yr</span>
        </span>
      </div>

      <div className="hud-sep" />

      {/* Roads */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">ROADS</span>
        <span className="font-mono text-xs font-bold text-white">
          {totalRoadKm.toFixed(2)} <span className="font-normal text-gcs-dim">km</span>
        </span>
      </div>

      <div className="hud-sep" />

      {/* Auto Accept % */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">AUTO-ACCEPT</span>
        <span className="font-mono text-xs font-bold text-gcs-green">{autoAcceptPct}%</span>
      </div>

      <div className="hud-sep" />

      {/* IoU Score — sourced from data/metrics.js, single source of truth */}
      <div className="flex items-center gap-2 px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">IoU</span>
        <span className="font-mono text-xs font-bold text-gcs-purple">
          {(buildingSegmentation.iou / 100).toFixed(3)}
        </span>
      </div>

      <div className="flex-1 min-w-[20px]" />

      {/* Live UTC Clock */}
      <div className="px-3 shrink-0">
        <span className="font-mono text-xs text-gcs-dim">{timeStr}</span>
      </div>

      {/* Region / Village Picker Button */}
      <button
        id="hud-region-picker-btn"
        onClick={onOpenVillagePicker}
        className="shrink-0 flex items-center gap-1.5 font-mono text-xs font-bold tracking-widest px-3 py-1 border border-gcs-border text-gcs-dim hover:text-gcs-green hover:border-gcs-green bg-transparent transition-colors ml-2"
        style={{ height: '28px', letterSpacing: '0.1em' }}
        title="Change Demo Region"
      >
        <ImageIcon className="w-3.5 h-3.5" />
        REGIONS
      </button>

      {/* ENGAGE / DISENGAGE Button */}
      <button
        onClick={onEngageToggle}
        className={`engage-btn shrink-0 font-mono text-xs font-bold tracking-widest px-4 py-1 border text-gcs-green border-gcs-green bg-transparent ml-2 ${
          engaged ? 'active' : ''
        }`}
        style={{ height: '28px', letterSpacing: '0.15em' }}
      >
        {engaged ? '⏹ DISENGAGE' : '▶ ENGAGE'}
      </button>
    </div>
  );
}
