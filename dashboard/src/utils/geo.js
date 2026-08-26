// Geographic bounding box for the 1000x1000 px tile (Nashik district demo area)
export const GEO_BOUNDS = {
  south: 19.9900,
  west:  73.7700,
  north: 20.0050,
  east:  73.7850,
};

export const TILE_W = 1000;
export const TILE_H = 1000;

export const ZONE_CENTER = [
  (GEO_BOUNDS.south + GEO_BOUNDS.north) / 2,
  (GEO_BOUNDS.west  + GEO_BOUNDS.east)  / 2
];

export const ZONE_BOUNDS = [
  [GEO_BOUNDS.south, GEO_BOUNDS.west],
  [GEO_BOUNDS.north, GEO_BOUNDS.east]
];

/**
 * Convert pixel coordinates (x right, y down from top-left)
 * to Leaflet [lat, lng] within the geo bounding box.
 */
export function pxToLatLng(px_x, px_y) {
  const lng = GEO_BOUNDS.west  + (px_x / TILE_W) * (GEO_BOUNDS.east  - GEO_BOUNDS.west);
  const lat = GEO_BOUNDS.north - (px_y / TILE_H) * (GEO_BOUNDS.north - GEO_BOUNDS.south);
  return [lat, lng];
}

/**
 * Roof material color mapping
 */
export function roofColor(mat) {
  switch (mat) {
    case 'RCC (Concrete)': return '#ff3355'; // Crimson
    case 'Tiled':          return '#ffaa00'; // Amber
    case 'Tin/Metal Sheet':return '#00d4ff'; // Cyan
    case 'Other':          return '#a855f7'; // Purple
    default:               return '#4a6075';
  }
}

/**
 * Tax heatmap color scale (cyan -> yellow -> crimson)
 */
export function taxColor(taxInr, maxTax) {
  const t = Math.min(taxInr / (maxTax || 1), 1);
  if (t < 0.5) {
    const r = Math.round(0   + t * 2 * 255);
    const g = Math.round(212 + t * 2 * (170 - 212));
    const b = Math.round(255 + t * 2 * (0 - 255));
    return `rgb(${r},${g},${b})`;
  } else {
    const t2 = (t - 0.5) * 2;
    const r = Math.round(255);
    const g = Math.round(170 - t2 * 170);
    const b = 0;
    return `rgb(${r},${g},${b})`;
  }
}

/**
 * Confidence score color mapping
 */
export function confColor(prob) {
  if (prob >= 0.8) return '#00ff88'; // Green
  if (prob >= 0.6) return '#ffaa00'; // Amber
  return '#ff3355';                  // Crimson
}


export function bboxToLatLngBounds(bbox = { x0: 0, x1: TILE_W, y0: 0, y1: TILE_H }) {
  const [south, west] = pxToLatLng(bbox.x0, bbox.y1);
  const [north, east] = pxToLatLng(bbox.x1, bbox.y0);
  return [[south, west], [north, east]];
}
