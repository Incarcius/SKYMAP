import React, { useState, useMemo, useEffect } from 'react';
import { TargetIcon, XIcon, CheckIcon, ShieldAlertIcon, BarChartIcon, ListIcon } from './Icons';
import { confColor, roofColor, pxToLatLng } from '../utils/geo';
import { getFlagReasons, getRecommendedAction } from '../utils/flagReasons';

const SEVERITY_COLOR = {
  high: '#ff3355',
  medium: '#ffaa00',
  info: '#00d4ff',
};

function WhyFlaggedSection({ building }) {
  const priority = building.review_priority || '';
  const isFlagged = priority && !priority.startsWith('LOW');
  if (!isFlagged) return null;

  const reasons = getFlagReasons(building);
  const action = getRecommendedAction(building);

  return (
    <div className="border-t border-gcs-border pt-1.5 space-y-1.5">
      <div className="flex items-center gap-1.5">
        <ShieldAlertIcon className="w-3 h-3 text-gcs-amber" />
        <span className="font-mono text-[10px] font-bold tracking-wider text-gcs-amber">
          WHY THIS PROPERTY WAS FLAGGED
        </span>
      </div>

      <div className="space-y-1.5">
        {reasons.map(r => (
          <div key={r.id} className="pl-1 border-l-2" style={{ borderColor: SEVERITY_COLOR[r.severity] }}>
            <div className="font-mono text-[10px] font-bold" style={{ color: SEVERITY_COLOR[r.severity] }}>
              {r.title}
            </div>
            <div className="font-mono text-[10px] text-gcs-dim leading-tight">{r.detail}</div>
          </div>
        ))}
      </div>

      {action && (
        <div className="mt-1.5 border border-gcs-amber/30 bg-gcs-amber/5 px-2 py-1.5">
          <div className="font-mono text-[9px] font-bold tracking-wider text-gcs-amber mb-0.5">
            RECOMMENDED ACTION
          </div>
          <div className="font-mono text-[10px] text-gcs-text leading-tight">{action}</div>
        </div>
      )}

      <div className="font-mono text-[9px] text-slate-600 italic">
        Rule-based explanation derived from model outputs.
      </div>
    </div>
  );
}

function GlobalAnalyticsTab({ buildings }) {
  // Aggregate stats
  const stats = useMemo(() => {
    let totalTax = 0;
    let totalSolar = 0;
    let totalArea = 0;
    const materialCounts = {
      'RCC (Concrete)': 0,
      'Tile': 0,
      'Tin/Metal': 0,
      'Other': 0
    };

    buildings.forEach(b => {
      totalTax += b.estimated_annual_tax_inr || 0;
      totalSolar += b.estimated_solar_kwp || 0;
      totalArea += b.area_m2 || 0;
      
      const mat = b.roof_material;
      if (mat === 'RCC (Concrete)') materialCounts['RCC (Concrete)']++;
      else if (mat === 'Tile') materialCounts['Tile']++;
      else if (mat === 'Tin/Metal') materialCounts['Tin/Metal']++;
      else materialCounts['Other']++;
    });

    return { totalTax, totalSolar, totalArea, materialCounts };
  }, [buildings]);

  const totalBldgs = buildings.length || 1;

  return (
    <div className="px-3 py-2.5 space-y-4 overflow-y-auto custom-scrollbar">
      {/* Overview Cards */}
      <div className="grid grid-cols-2 gap-2">
        <div className="border border-gcs-border bg-slate-900/40 p-2 rounded flex flex-col justify-center items-center">
          <div className="font-mono text-[9px] text-gcs-dim tracking-widest mb-1 text-center">TOTAL REVENUE EST</div>
          <div className="font-mono text-sm font-bold text-gcs-amber">₹{Math.round(stats.totalTax).toLocaleString('en-IN')}</div>
        </div>
        <div className="border border-gcs-border bg-slate-900/40 p-2 rounded flex flex-col justify-center items-center">
          <div className="font-mono text-[9px] text-gcs-dim tracking-widest mb-1 text-center">TOTAL SOLAR CAP</div>
          <div className="font-mono text-sm font-bold text-gcs-green">{stats.totalSolar.toFixed(1)} kWp</div>
        </div>
        <div className="border border-gcs-border bg-slate-900/40 p-2 rounded flex flex-col justify-center items-center">
          <div className="font-mono text-[9px] text-gcs-dim tracking-widest mb-1 text-center">TOTAL BUILT-UP AREA</div>
          <div className="font-mono text-sm font-bold text-white">{Math.round(stats.totalArea).toLocaleString()} m²</div>
        </div>
        <div className="border border-gcs-border bg-slate-900/40 p-2 rounded flex flex-col justify-center items-center">
          <div className="font-mono text-[9px] text-gcs-dim tracking-widest mb-1 text-center">DETECTED BUILDINGS</div>
          <div className="font-mono text-sm font-bold text-white">{buildings.length}</div>
        </div>
      </div>

      {/* Material Breakdown */}
      <div>
        <div className="font-mono text-[10px] font-bold text-gcs-cyan mb-2 tracking-widest border-b border-gcs-border pb-1">
          ROOF MATERIAL DISTRIBUTION
        </div>
        <div className="space-y-2">
          {Object.entries(stats.materialCounts).map(([mat, count]) => {
            const pct = (count / totalBldgs) * 100;
            return (
              <div key={mat}>
                <div className="flex justify-between font-mono text-[10px] mb-0.5">
                  <span style={{ color: roofColor(mat) }}>{mat}</span>
                  <span className="text-gcs-dim">{count} ({pct.toFixed(1)}%)</span>
                </div>
                <div className="h-1.5 w-full bg-slate-900 overflow-hidden rounded">
                  <div className="h-full" style={{ width: `${pct}%`, backgroundColor: roofColor(mat) }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

export default function RightSidebar({
  buildings,
  selectedBuilding,
  onCloseInspector,
  onAccept,
  onReject,
  onRowClick,
  triageMode,
}) {
  const [activeTab, setActiveTab] = useState('queue'); // 'inspector', 'queue', 'analytics'

  // If a building gets selected from outside (e.g. map click), auto-switch to inspector
  useEffect(() => {
    if (selectedBuilding) {
      setActiveTab('inspector');
    }
  }, [selectedBuilding]);

  // Filter queue buildings: in triage mode show ONLY flagged; otherwise exclude LOW + accepted/rejected
  const queueBuildings = buildings
    .filter(b => {
      if (b.status === 'accepted' || b.status === 'rejected') return false;
      if (triageMode) return !b.review_priority?.startsWith('LOW');
      return !b.review_priority?.startsWith('LOW');
    })
    .sort((a, b) => (a.mean_pixel_probability || 0) - (b.mean_pixel_probability || 0));

  // Compute inspector coordinates for selected building
  let centerLat = 0;
  let centerLng = 0;
  if (selectedBuilding && selectedBuilding.polygon_px && selectedBuilding.polygon_px.length > 0) {
    const latlngs = selectedBuilding.polygon_px.map(p => pxToLatLng(p[0], p[1]));
    centerLat = latlngs.reduce((s, ll) => s + ll[0], 0) / latlngs.length;
    centerLng = latlngs.reduce((s, ll) => s + ll[1], 0) / latlngs.length;
  }

  const b = selectedBuilding;
  const prob = b?.mean_pixel_probability || 0;
  const cCol = confColor(prob);
  const taxFmt = b ? (b.estimated_annual_tax_inr || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 }) : '0';
  const status = b?.status || 'unverified';
  const rawPriority = b?.review_priority || '';
  const priority = rawPriority.includes('HIGH') ? 'HIGH' : rawPriority.includes('MEDIUM') ? 'MED' : 'LOW';
  const priColor = priority === 'HIGH' ? '#ff3355' : priority === 'MED' ? '#ffaa00' : '#00ff88';

  return (
    <aside id="right-sidebar" className="absolute right-4 top-16 bottom-12 w-84 flex flex-col gap-0 z-[1000] select-none pointer-events-auto bg-slate-950/90 border border-gcs-border shadow-lg rounded">
      
      {/* ── TABS ────────────────────────────────────────────────────────────── */}
      <div className="flex border-b border-gcs-border bg-slate-900/60 shrink-0">
        <button
          onClick={() => setActiveTab('inspector')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 font-mono text-[10px] tracking-wider transition-colors ${
            activeTab === 'inspector' ? 'text-gcs-crimson border-b-2 border-gcs-crimson bg-gcs-crimson/10 font-bold' : 'text-gcs-dim hover:text-white'
          }`}
        >
          <TargetIcon className="w-3.5 h-3.5" />
          INSPECTOR
        </button>
        <button
          onClick={() => setActiveTab('queue')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 font-mono text-[10px] tracking-wider transition-colors ${
            activeTab === 'queue' ? 'text-gcs-amber border-b-2 border-gcs-amber bg-gcs-amber/10 font-bold' : 'text-gcs-dim hover:text-white'
          }`}
        >
          <ListIcon className="w-3.5 h-3.5" />
          QUEUE
          {queueBuildings.length > 0 && (
            <span className="ml-1 bg-gcs-amber text-black px-1 rounded-sm text-[9px]">{queueBuildings.length}</span>
          )}
        </button>
        <button
          onClick={() => setActiveTab('analytics')}
          className={`flex-1 flex items-center justify-center gap-1.5 py-2.5 font-mono text-[10px] tracking-wider transition-colors ${
            activeTab === 'analytics' ? 'text-gcs-cyan border-b-2 border-gcs-cyan bg-gcs-cyan/10 font-bold' : 'text-gcs-dim hover:text-white'
          }`}
        >
          <BarChartIcon className="w-3.5 h-3.5" />
          ANALYTICS
        </button>
      </div>

      {/* ── TAB CONTENT ─────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden flex flex-col min-h-0">
        
        {/* INSPECTOR TAB */}
        {activeTab === 'inspector' && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {b ? (
              <div className="px-3 py-2.5 space-y-2 overflow-y-auto custom-scrollbar">
                <div className="flex justify-between items-center">
                  <div className="font-mono text-sm font-bold text-white">BLDG #{b.id}</div>
                  <span
                    className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded uppercase"
                    style={{
                      color: status === 'accepted' ? '#00ff88' : status === 'rejected' ? '#ff3355' : '#ffaa00',
                      border: `1px solid ${status === 'accepted' ? '#00ff8844' : status === 'rejected' ? '#ff335544' : '#ffaa0044'}`,
                      backgroundColor: status === 'accepted' ? '#00ff8811' : status === 'rejected' ? '#ff335511' : '#ffaa0011',
                    }}
                  >
                    {status}
                  </span>
                </div>

                {/* AI Confidence */}
                <div>
                  <div className="flex justify-between mb-1">
                    <span className="font-mono text-[11px] text-gcs-dim">AI CONFIDENCE</span>
                    <span className="font-mono text-[11px] font-bold" style={{ color: cCol }}>
                      {(prob * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="conf-bar">
                    <div
                      className="conf-fill"
                      style={{ width: `${(prob * 100).toFixed(1)}%`, backgroundColor: cCol }}
                    />
                  </div>
                </div>

                {/* Metric grid */}
                <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 pt-1">
                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-gcs-dim">PRIORITY</span>
                    <span className="font-mono text-xs font-bold" style={{ color: priColor }}>
                      {priority}
                    </span>
                  </div>

                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-gcs-dim">BUILT-UP AREA</span>
                    <span className="font-mono text-xs text-white">{b.area_m2} m²</span>
                  </div>

                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-gcs-dim">ROOF MATERIAL</span>
                    <span className="font-mono text-xs font-semibold" style={{ color: roofColor(b.roof_material) }}>
                      {b.roof_material}
                    </span>
                  </div>

                  <div className="flex flex-col">
                    <span className="font-mono text-[10px] text-gcs-dim">SOLAR POTENTIAL</span>
                    <span className="font-mono text-xs text-gcs-green">{b.estimated_solar_kwp} kWp</span>
                  </div>

                  <div className="flex flex-col col-span-2">
                    <span className="font-mono text-[10px] text-gcs-dim">ESTIMATED ANNUAL TAX</span>
                    <span className="font-mono text-xs font-bold text-gcs-amber">₹{taxFmt} / year</span>
                  </div>
                </div>

                {/* Regularization + Roof Confidence */}
                {(b.regularization_method || b.roof_confidence != null) && (
                  <div className="border-t border-gcs-border pt-1.5 space-y-1.5">
                    {b.regularization_method && (
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[10px] text-gcs-dim">REGULARIZATION</span>
                        <span
                          className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded uppercase"
                          style={{
                            color: b.regularization_method === 'snapped_to_rect' ? '#00d4ff' : '#a855f7',
                            border: `1px solid ${
                              b.regularization_method === 'snapped_to_rect' ? '#00d4ff44' : '#a855f744'
                            }`,
                            backgroundColor: b.regularization_method === 'snapped_to_rect'
                              ? '#00d4ff11'
                              : '#a855f711',
                          }}
                        >
                          {b.regularization_method === 'snapped_to_rect' ? 'RECT-SNAP' : 'EDGE-STRAIGHT'}
                        </span>
                      </div>
                    )}
                    {b.roof_confidence != null && (
                      <div>
                        <div className="flex justify-between mb-0.5">
                          <span className="font-mono text-[10px] text-gcs-dim">ROOF CONF (KMeans)</span>
                          <span
                            className="font-mono text-[10px] font-bold"
                            style={{ color: confColor(b.roof_confidence) }}
                          >
                            {(b.roof_confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="conf-bar">
                          <div
                            className="conf-fill"
                            style={{
                              width: `${(b.roof_confidence * 100).toFixed(0)}%`,
                              backgroundColor: confColor(b.roof_confidence),
                              opacity: 0.7,
                            }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* NEW: Why Was This Building Flagged? */}
                <WhyFlaggedSection building={b} />

                {/* Coordinates */}
                <div className="border-t border-gcs-border pt-1.5 flex justify-between font-mono text-[10px] text-gcs-cyan">
                  <span>LAT {centerLat.toFixed(6)}</span>
                  <span>LON {centerLng.toFixed(6)}</span>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center p-4 text-center text-gcs-dim font-mono text-xs">
                <TargetIcon className="w-8 h-8 mb-2 opacity-30" />
                NO TARGET SELECTED
                <div className="text-[10px] text-slate-600 mt-1">
                  Click a map polygon or a queue card
                </div>
              </div>
            )}
          </div>
        )}

        {/* QUEUE TAB */}
        {activeTab === 'queue' && (
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Triage Mode Banner */}
            {triageMode && (
              <div className="triage-banner shrink-0 m-2 rounded">
                <span className="w-1.5 h-1.5 rounded-full bg-gcs-amber animate-pulse inline-block mr-1.5" />
                TRIAGE MODE ACTIVE — FLAGGED ONLY
              </div>
            )}
            {/* Scrollable Compact Card List */}
            <div className="flex-1 overflow-y-auto p-2 space-y-2 custom-scrollbar">
              {queueBuildings.length === 0 ? (
                <div className="py-8 text-center text-gcs-green font-mono text-xs flex flex-col items-center">
                  <CheckIcon className="w-6 h-6 mb-2" />
                  ALL FLAGGED BUILDINGS VERIFIED
                </div>
              ) : (
                queueBuildings.map(item => {
                  const itemProb = item.mean_pixel_probability || 0;
                  const itemColor = confColor(itemProb);
                  const isHigh = item.review_priority?.includes('HIGH');
                  const tagColor = isHigh ? '#ff3355' : '#ffaa00';
                  const isSelected = selectedBuilding?.id === item.id;

                  return (
                    <div
                      key={`qcard-${item.id}`}
                      onClick={() => onRowClick(item)}
                      className={`p-2 rounded border transition-all cursor-pointer ${
                        isSelected
                          ? 'bg-gcs-cyan/10 border-gcs-cyan'
                          : 'bg-slate-900/80 border-gcs-border hover:border-slate-600'
                      }`}
                    >
                      {/* Card Top Row: ID, Priority Tag, Confidence, Action Buttons */}
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-xs font-bold text-gcs-cyan">
                            #{item.id}
                          </span>
                          <span
                            className="font-mono text-[9px] font-bold px-1 rounded"
                            style={{
                              color: tagColor,
                              border: `1px solid ${tagColor}44`,
                              backgroundColor: `${tagColor}11`,
                            }}
                          >
                            {isHigh ? 'HIGH' : 'MED'}
                          </span>
                          <span
                            className="font-mono text-[10px] font-bold"
                            style={{ color: itemColor }}
                          >
                            {(itemProb * 100).toFixed(0)}% CONF
                          </span>
                        </div>

                        {/* Quick Accept/Reject */}
                        <div className="flex gap-1" onClick={e => e.stopPropagation()}>
                          <button
                            onClick={() => { onAccept(item.id); setActiveTab('queue'); }}
                            className="p-1 border border-gcs-green text-gcs-green hover:bg-gcs-green hover:text-black transition-colors rounded"
                            title="Accept Verification"
                          >
                            <CheckIcon className="w-3 h-3" />
                          </button>
                          <button
                            onClick={() => { onReject(item.id); setActiveTab('queue'); }}
                            className="p-1 border border-gcs-crimson text-gcs-crimson hover:bg-gcs-crimson hover:text-white transition-colors rounded"
                            title="Reject Verification"
                          >
                            <XIcon className="w-3 h-3" />
                          </button>
                        </div>
                      </div>

                      {/* Card Bottom Row: Area & Material */}
                      <div className="flex justify-between text-[10px] font-mono text-slate-400">
                        <span>AREA: {item.area_m2} m²</span>
                        <span style={{ color: roofColor(item.roof_material) }}>
                          {item.roof_material}
                        </span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        )}

        {/* ANALYTICS TAB */}
        {activeTab === 'analytics' && (
          <GlobalAnalyticsTab buildings={buildings} />
        )}
      </div>
    </aside>
  );
}
