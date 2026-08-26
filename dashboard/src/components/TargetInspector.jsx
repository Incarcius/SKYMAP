import React from 'react';
import { TargetIcon, XIcon } from './Icons';
import { confColor, roofColor, pxToLatLng } from '../utils/geo';

export default function TargetInspector({ selectedTarget, onClose }) {
  if (!selectedTarget) return null;

  const isBuilding = selectedTarget.type === 'building' || selectedTarget.id !== undefined;
  const b = isBuilding ? (selectedTarget.data || selectedTarget) : null;

  let centerLat = 0;
  let centerLng = 0;

  if (b && b.polygon_px && b.polygon_px.length > 0) {
    const latlngs = b.polygon_px.map(p => pxToLatLng(p[0], p[1]));
    centerLat = latlngs.reduce((s, ll) => s + ll[0], 0) / latlngs.length;
    centerLng = latlngs.reduce((s, ll) => s + ll[1], 0) / latlngs.length;
  }

  const prob = b?.mean_pixel_probability || 0;
  const cCol = confColor(prob);
  const taxFmt = (b?.estimated_annual_tax_inr || 0).toLocaleString('en-IN', { maximumFractionDigits: 0 });
  
  const status = b?.status || 'unverified';
  const rawPriority = b?.review_priority || '';
  const priority = rawPriority.includes('HIGH') ? 'HIGH' : rawPriority.includes('MEDIUM') ? 'MED' : 'LOW';
  const priColor = priority === 'HIGH' ? '#ff3355' : priority === 'MED' ? '#ffaa00' : '#00ff88';

  return (
    <div
      id="inspector"
      className="gcs-panel absolute right-4 top-4 z-[1000] w-[230px] p-0 transition-opacity duration-300 select-none"
    >
      {/* Panel Header */}
      <div className="flex items-center gap-2 px-3 py-2 border-b border-gcs-border justify-between">
        <div className="flex items-center gap-2">
          <TargetIcon className="w-4 h-4 text-gcs-crimson" />
          <span className="font-mono text-xs font-bold tracking-widest text-gcs-crimson">
            INSPECTOR
          </span>
        </div>
        <button
          onClick={onClose}
          className="font-mono text-xs text-gcs-dim hover:text-white transition-colors"
        >
          <XIcon className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Body */}
      <div className="px-3 py-2 space-y-2">
        <div className="font-mono text-xs text-gcs-dim">TARGET</div>
        <div className="font-mono text-sm font-bold text-white mb-2">
          BLDG #{b.id}
        </div>

        <div className="space-y-1">
          {/* AI Confidence */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">AI·CONF</span>
            <span className="font-mono text-xs font-bold" style={{ color: cCol }}>
              {(prob * 100).toFixed(1)}%
            </span>
          </div>
          <div className="conf-bar mb-2">
            <div
              className="conf-fill"
              style={{ width: `${(prob * 100).toFixed(1)}%`, backgroundColor: cCol }}
            />
          </div>

          {/* Status */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">STATUS</span>
            <span
              className="font-mono text-xs font-bold uppercase"
              style={{
                color: status === 'accepted' ? '#00ff88' : status === 'rejected' ? '#ff3355' : '#ffaa00',
              }}
            >
              {status}
            </span>
          </div>

          {/* Priority */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">PRIORITY</span>
            <span className="font-mono text-xs font-bold" style={{ color: priColor }}>
              {priority}
            </span>
          </div>

          {/* Built-up Area */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">AREA</span>
            <span className="font-mono text-xs text-white">{b.area_m2} m²</span>
          </div>

          {/* Material */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">MATERIAL</span>
            <span className="font-mono text-xs" style={{ color: roofColor(b.roof_material) }}>
              {b.roof_material}
            </span>
          </div>

          {/* Solar Potential */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">SOLAR</span>
            <span className="font-mono text-xs text-gcs-green">{b.estimated_solar_kwp} kWp</span>
          </div>

          {/* Annual Tax */}
          <div className="flex justify-between">
            <span className="font-mono text-xs text-gcs-dim">EST·TAX</span>
            <span className="font-mono text-xs text-gcs-amber">₹{taxFmt}/yr</span>
          </div>
        </div>

        {/* Lat / Lng Coordinates */}
        <div className="border-t border-gcs-border mt-2 pt-2">
          <div className="font-mono text-xs text-gcs-dim mb-1">COORDINATES</div>
          <div className="font-mono text-xs text-gcs-cyan">LAT {centerLat.toFixed(6)}</div>
          <div className="font-mono text-xs text-gcs-cyan">LNG {centerLng.toFixed(6)}</div>
        </div>
      </div>
    </div>
  );
}
