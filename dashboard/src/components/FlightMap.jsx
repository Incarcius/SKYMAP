import React, { useState, useEffect } from 'react';
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
  Polygon,
  Polyline,
  Rectangle,
  useMap,
  useMapEvents,
} from 'react-leaflet';
import {
  ZONE_BOUNDS,
  pxToLatLng,
  roofColor,
  taxColor,
  confColor,
} from '../utils/geo';
import { IMG_B64 } from '../data/rgb_b64';

/**
 * Event listener hook component for mousemove cursor tracking
 */
function MapEventListener({ onCursorMove }) {
  useMapEvents({
    mousemove: e => {
      if (onCursorMove) onCursorMove(e.latlng);
    },
  });
  return null;
}

/**
 * Controller component inside MapContainer to react to flyTarget, zoom lock & engaged props.
 */
function MapController({ engaged, flyTarget, setIsAnimating }) {
  const map = useMap();

  // Fly down on ENGAGE toggle and manage animation state & zoom lock
  useEffect(() => {
    setIsAnimating(true); // Start animation state
    if (engaged) {
      map.flyToBounds(ZONE_BOUNDS, { padding: [10, 10], animate: true, duration: 1.5 });
      const timer = setTimeout(() => {
        map.setMinZoom(map.getZoom());
        setIsAnimating(false); // End animation state
      }, 1600);
      return () => clearTimeout(timer);
    } else {
      map.setMinZoom(3);
      map.flyTo([20.5937, 78.9629], 5, { duration: 1.5 });
      const timer = setTimeout(() => {
        setIsAnimating(false); // End animation state
      }, 1600);
      return () => clearTimeout(timer);
    }
  }, [engaged, map, setIsAnimating]);

  // Fly to specific building when flyTarget changes
  useEffect(() => {
    if (flyTarget && flyTarget.bounds) {
      map.flyToBounds(flyTarget.bounds, {
        padding: [40, 40],
        animate: true,
        duration: 1.2,
        maxZoom: 20,
      });
    }
  }, [flyTarget, map]);

  return null;
}

export default function FlightMap({
  buildings,
  roads,
  water,
  layerVis,
  thematicMode,
  engaged,
  flyTarget,
  selectedBuilding,
  onSelectBuilding,
  onCursorMove,
  triageMode,
}) {
  const [isAnimating, setIsAnimating] = useState(false);

  const maxTax = Math.max(...buildings.map(b => b.estimated_annual_tax_inr || 0), 1);

  // Dynamic building polygon styling based on state & thematic mode
  const getBldgStyle = b => {
    // Triage mode: dim non-flagged buildings so flagged ones stand out
    const isLowPriority = b.review_priority?.startsWith('LOW');
    if (triageMode && isLowPriority && b.status !== 'accepted' && b.status !== 'rejected') {
      return {
        color: '#4a6075',
        fillColor: '#4a6075',
        fillOpacity: 0.08,
        weight: 0.5,
      };
    }

    if (b.status === 'accepted') {
      return {
        color: '#00ff88',
        fillColor: '#00ff88',
        fillOpacity: 0.45,
        weight: 2,
      };
    }
    if (b.status === 'rejected') {
      return {
        color: '#ff3355',
        fillColor: '#ff3355',
        fillOpacity: 0.15,
        weight: 1,
      };
    }

    let col = '#4a6075';
    if (thematicMode === 'tax') {
      col = taxColor(b.estimated_annual_tax_inr || 0, maxTax);
    } else if (thematicMode === 'solar') {
      col = b.estimated_solar_kwp > 30 ? '#00ff88' : '#00d4ff';
    } else if (thematicMode === 'conf') {
      col = confColor(b.mean_pixel_probability || 0);
    } else {
      col = roofColor(b.roof_material);
    }

    const isSelected = selectedBuilding && selectedBuilding.id === b.id;

    return {
      color: col,
      fillColor: col,
      fillOpacity: isSelected ? 0.55 : 0.25,
      weight: isSelected ? 3 : 1.5,
    };
  };

  // Compute reticle bounds for selected building
  let reticleBounds = null;
  if (selectedBuilding && selectedBuilding.polygon_px) {
    const latlngs = selectedBuilding.polygon_px.map(p => pxToLatLng(p[0], p[1]));
    const lats = latlngs.map(ll => ll[0]);
    const lngs = latlngs.map(ll => ll[1]);
    const s = Math.min(...lats), n = Math.max(...lats);
    const w = Math.min(...lngs), e = Math.max(...lngs);
    const dLat = (n - s) * 0.4 + 0.000015;
    const dLng = (e - w) * 0.4 + 0.000015;
    reticleBounds = [
      [s - dLat, w - dLng],
      [n + dLat, e + dLng],
    ];
  }

  // Outer and inner rings for the Spotlight Mask (dimming surroundings when engaged)
  const outerRing = [[-90, -180], [90, -180], [90, 180], [-90, 180]];
  const innerRing = [
    [ZONE_BOUNDS[0][0], ZONE_BOUNDS[0][1]], // South-West
    [ZONE_BOUNDS[1][0], ZONE_BOUNDS[0][1]], // North-West
    [ZONE_BOUNDS[1][0], ZONE_BOUNDS[1][1]], // North-East
    [ZONE_BOUNDS[0][0], ZONE_BOUNDS[1][1]], // South-East
  ];

  return (
    <div id="map" className={`w-full h-full relative z-[1]${engaged || isAnimating ? ' map-engaged' : ''}`}>
      <MapContainer
        center={[20.5937, 78.9629]}
        zoom={5}
        minZoom={3}
        maxBounds={[[-90, -180], [90, 180]]}
        maxBoundsViscosity={1.0}
        zoomControl={true}
        className="w-full h-full"
      >
        <MapController engaged={engaged} flyTarget={flyTarget} setIsAnimating={setIsAnimating} />
        <MapEventListener onCursorMove={onCursorMove} />

        {/* Satellite Base Layer — grayscale applied via .map-engaged CSS when engaged */}
        <TileLayer
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          attribution="Esri World Imagery"
          maxZoom={21}
          noWrap={true}
        />

        {/* Spotlight Mask: Dims rest of Earth ONLY IF engaged && !isAnimating */}
        {engaged && !isAnimating && (
          <Polygon
            positions={[outerRing, innerRing]}
            pathOptions={{
              stroke: false,
              fillColor: '#000000',
              fillOpacity: 0.25,
            }}
          />
        )}

        {/* Target Zone Red Box: Render ONLY IF !engaged && !isAnimating */}
        {!engaged && !isAnimating && (
          <Rectangle
            bounds={ZONE_BOUNDS}
            pathOptions={{
              color: '#ff3355',
              weight: 2,
              dashArray: '6 4',
              fill: false,
              fillOpacity: 0,
              fillColor: 'transparent',
            }}
          />
        )}

        {/* Base64 Aerial Orthophoto: Render IF engaged || isAnimating for smooth zoom scaling */}
        {(engaged || isAnimating) && (
          <ImageOverlay
            url={IMG_B64}
            bounds={ZONE_BOUNDS}
            opacity={1.0}
          />
        )}

        {/* Tactical Boundary Frame: Render ONLY IF engaged && !isAnimating */}
        {engaged && !isAnimating && (
          <Rectangle
            bounds={ZONE_BOUNDS}
            pathOptions={{
              color: '#0ea5e9',
              weight: 6,
              fill: false,
              fillOpacity: 0,
              fillColor: 'transparent',
            }}
          />
        )}

        {/* Vector Waterbodies: Render ONLY IF engaged && !isAnimating */}
        {engaged && !isAnimating && layerVis.water && water.map((w, idx) => {
          if (!w.polygon_px || w.polygon_px.length < 3) return null;
          const latlngs = w.polygon_px.map(p => pxToLatLng(p[0], p[1]));
          return (
            <Polygon
              key={`water-${w.id || idx}`}
              positions={latlngs}
              pathOptions={{
                color: '#00d4ff',
                fillColor: '#00d4ff',
                fillOpacity: 0.25,
                weight: 1.5,
              }}
            />
          );
        })}

        {/* Vector Roads: Render ONLY IF engaged && !isAnimating */}
        {engaged && !isAnimating && layerVis.roads && roads.map((r, idx) => {
          if (!r.polyline_px || r.polyline_px.length < 2) return null;
          const latlngs = r.polyline_px.map(p => pxToLatLng(p[0], p[1]));
          return (
            <Polyline
              key={`road-${r.id || idx}`}
              positions={latlngs}
              pathOptions={{
                color: '#ffaa00',
                weight: 2.5,
                opacity: 0.9,
              }}
            />
          );
        })}

        {/* Vector Buildings: Render ONLY IF engaged && !isAnimating */}
        {engaged && !isAnimating && layerVis.buildings && buildings.map((b, idx) => {
          if (!b.polygon_px || b.polygon_px.length < 3) return null;
          const latlngs = b.polygon_px.map(p => pxToLatLng(p[0], p[1]));
          const style = getBldgStyle(b);
          return (
            <Polygon
              key={`bldg-${b.id || idx}`}
              positions={latlngs}
              pathOptions={style}
              eventHandlers={{
                click: () => onSelectBuilding(b),
              }}
            />
          );
        })}

        {/* Reticle Overlay: Render ONLY IF !isAnimating */}
        {!isAnimating && reticleBounds && (
          <Rectangle
            bounds={reticleBounds}
            pathOptions={{
              color: '#00ff88',
              weight: 2,
              dashArray: '4 4',
              fill: false,
              fillOpacity: 0,
              fillColor: 'transparent',
              className: 'reticle-pulse',
            }}
          />
        )}
      </MapContainer>
    </div>
  );
}
