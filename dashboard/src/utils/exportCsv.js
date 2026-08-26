import { pxToLatLng } from './geo';
import { getFlagReasons, getRecommendedAction } from './flagReasons';

function escapeCsv(value) {
  if (value === null || value === undefined) return '';
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replace(/"/g, '""')}"`;
  return text;
}

function getBuildingCenter(b, bbox) {
  if (!b?.polygon_px?.length) return { x: '', y: '', lat: '', lng: '' };
  const points = b.polygon_px.map(([x, y]) => ({
    x: ((x - bbox.x0) / (bbox.x1 - bbox.x0)) * 1000,
    y: ((y - bbox.y0) / (bbox.y1 - bbox.y0)) * 1000,
  }));
  const x = points.reduce((sum, p) => sum + p.x, 0) / points.length;
  const y = points.reduce((sum, p) => sum + p.y, 0) / points.length;
  const [lat, lng] = pxToLatLng(x, y);
  return { x: x.toFixed(2), y: y.toFixed(2), lat: lat.toFixed(6), lng: lng.toFixed(6) };
}

export function exportBuildingResultsCsv(buildings, activeVillage) {
  const bbox = activeVillage?.bbox || { x0: 0, x1: 1000, y0: 0, y1: 1000 };
  const headers = [
    'survey_area', 'building_id', 'status', 'review_priority', 'confidence_pct',
    'area_m2', 'roof_material', 'roof_confidence_pct', 'solar_kwp',
    'estimated_annual_tax_inr', 'regularization_method', 'center_x_px', 'center_y_px',
    'latitude', 'longitude', 'flag_reasons', 'recommended_action',
  ];
  const rows = buildings.map(b => {
    const center = getBuildingCenter(b, bbox);
    const reasons = getFlagReasons(b).map(r => r.title).join(' | ');
    return [
      activeVillage?.name || 'Survey Area', b.id, b.status || 'unverified', b.review_priority || '',
      b.mean_pixel_probability != null ? (b.mean_pixel_probability * 100).toFixed(1) : '',
      b.area_m2 ?? '', b.roof_material || '',
      b.roof_confidence != null ? (b.roof_confidence * 100).toFixed(1) : '',
      b.estimated_solar_kwp ?? '', b.estimated_annual_tax_inr ?? '', b.regularization_method || '',
      center.x, center.y, center.lat, center.lng, reasons, getRecommendedAction(b) || '',
    ];
  });
  const csv = [headers, ...rows].map(row => row.map(escapeCsv).join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const safeArea = (activeVillage?.id || 'survey-area').replace(/[^a-z0-9_-]+/gi, '-');
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const filename = `skymap-${safeArea}-results-${timestamp}.csv`;
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return { count: rows.length, filename };
}
