import React from 'react';
import { CheckIcon, XIcon } from './Icons';
import { confColor, roofColor } from '../utils/geo';

export default function TriageQueue({
  buildings,
  selectedBuildingId,
  onAccept,
  onReject,
  onRowClick,
}) {
  // Filter queue buildings: exclude auto-accepted LOW priority and already accepted buildings
  const queueBuildings = buildings
    .filter(
      b =>
        b.status !== 'accepted' &&
        b.status !== 'rejected' &&
        !b.review_priority?.startsWith('LOW')
    )
    .sort((a, b) => (a.mean_pixel_probability || 0) - (b.mean_pixel_probability || 0));

  const totalFlagged = queueBuildings.length;

  return (
    <div
      id="queue-panel"
      className="gcs-panel shrink-0 h-[25vh] w-full border-l-0 border-r-0 border-b-0 flex flex-col overflow-hidden z-[950]"
    >
      {/* Header Bar */}
      <div className="flex items-center gap-3 px-3 py-1 border-b border-gcs-border flex-shrink-0 h-[30px]">
        <svg width="10" height="10" viewBox="0 0 10 10">
          <rect width="10" height="10" fill="#ffaa00" opacity="0.9" />
        </svg>
        <span className="font-mono text-xs font-bold tracking-widest text-gcs-amber">
          TRIAGE QUEUE
        </span>
        <span className="font-mono text-xs text-gcs-dim">
          — {totalFlagged} pending review &nbsp;·&nbsp; Click row to GOTO target
        </span>
        <div className="flex-1" />
        <span className="font-mono text-xs text-gcs-dim">
          3-model ensemble &nbsp;·&nbsp; threshold=0.56
        </span>
      </div>

      {/* Table Container */}
      <div className="flex-1 overflow-y-auto overflow-x-auto">
        <table className="queue-table">
          <thead>
            <tr>
              <th>ACTION</th>
              <th>ID</th>
              <th>CMD</th>
              <th>AI·CONF</th>
              <th>PRIORITY</th>
              <th>AREA_M²</th>
              <th>MATERIAL</th>
              <th>SOLAR_kWp</th>
              <th>TAX_INR/yr</th>
            </tr>
          </thead>
          <tbody>
            {queueBuildings.length === 0 ? (
              <tr>
                <td colSpan="9" className="text-center py-6 text-gcs-green font-mono text-xs">
                  ✓ ALL FLAGGED BUILDINGS VERIFIED & ACCEPTED
                </td>
              </tr>
            ) : (
              queueBuildings.map(b => {
                const prob = b.mean_pixel_probability || 0;
                const cCol = confColor(prob);
                const isHigh = b.review_priority?.includes('HIGH');
                const priority = isHigh ? 'HIGH' : 'MED';
                const priColor = isHigh ? '#ff3355' : '#ffaa00';
                const taxFmt = (b.estimated_annual_tax_inr || 0).toLocaleString('en-IN', {
                  maximumFractionDigits: 0,
                });
                const isSelected = selectedBuildingId === b.id;

                return (
                  <tr
                    key={`q-${b.id}`}
                    className={isSelected ? 'selected' : ''}
                    onClick={() => onRowClick(b)}
                  >
                    {/* Action buttons */}
                    <td onClick={e => e.stopPropagation()}>
                      <div className="flex gap-1">
                        <button
                          onClick={() => onAccept(b.id)}
                          className="font-mono text-xs px-2 py-0.5 border border-gcs-green text-gcs-green hover:bg-gcs-green hover:text-black transition-colors flex items-center gap-1"
                          title="Accept detection"
                        >
                          <CheckIcon className="w-3 h-3" />
                        </button>
                        <button
                          onClick={() => onReject(b.id)}
                          className="font-mono text-xs px-2 py-0.5 border border-gcs-crimson text-gcs-crimson hover:bg-gcs-crimson hover:text-white transition-colors flex items-center gap-1"
                          title="Reject detection"
                        >
                          <XIcon className="w-3 h-3" />
                        </button>
                      </div>
                    </td>

                    <td className="text-gcs-cyan">#{b.id}</td>
                    <td className="text-gcs-dim">VERIFY</td>
                    <td style={{ color: cCol }}>{(prob * 100).toFixed(1)}%</td>
                    <td style={{ color: priColor }}>{priority}</td>
                    <td>{b.area_m2}</td>
                    <td style={{ color: roofColor(b.roof_material) }}>
                      {(b.roof_material || '')
                        .replace('(Concrete)', '(RCC)')
                        .replace('Tin/Metal Sheet', 'Tin')}
                    </td>
                    <td className="text-gcs-green">{b.estimated_solar_kwp}</td>
                    <td className="text-gcs-amber">₹{taxFmt}</td>
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
