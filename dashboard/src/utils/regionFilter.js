function pointInBbox([x, y], bbox) {
  return x >= bbox.x0 && x <= bbox.x1 && y >= bbox.y0 && y <= bbox.y1;
}

function polylineCoverage(points, bbox) {
  if (!points?.length) return 0;
  return points.filter(p => pointInBbox(p, bbox)).length / points.length;
}

function bboxOfPoints(points) {
  const xs = points.map(p => p[0]);
  const ys = points.map(p => p[1]);
  return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}

function bboxIntersects(a, b) {
  return a.x0 <= b.x1 && a.x1 >= b.x0 && a.y0 <= b.y1 && a.y1 >= b.y0;
}

function polygonCenter(points) {
  if (!points?.length) return null;
  return [
    points.reduce((s, p) => s + p[0], 0) / points.length,
    points.reduce((s, p) => s + p[1], 0) / points.length,
  ];
}

export function filterBuildingsByRegion(buildings, bbox) {
  return buildings.filter(b => {
    const points = b.polygon_px || [];
    if (points.length < 3) return false;
    const shape = bboxOfPoints(points);
    const center = polygonCenter(points);
    return bboxIntersects(shape, bbox) && pointInBbox(center, bbox);
  });
}

export function filterRoadsByRegion(roads, bbox) {
  return roads.filter(r => polylineCoverage(r.polyline_px || [], bbox) >= 0.15);
}

export function filterWaterByRegion(water, bbox) {
  return water.filter(w => {
    const points = w.polygon_px || [];
    if (points.length < 3) return false;
    const center = polygonCenter(points);
    return pointInBbox(center, bbox) && bboxIntersects(bboxOfPoints(points), bbox);
  });
}

export function filterAllByRegion(buildings, roads, water, bbox) {
  return {
    buildings: filterBuildingsByRegion(buildings, bbox),
    roads: filterRoadsByRegion(roads, bbox),
    water: filterWaterByRegion(water, bbox),
  };
}
