import React, { useState } from 'react';
import TelemetryHUD from './components/TelemetryHUD';
import FlightMap from './components/FlightMap';
import LayerControls from './components/LayerControls';
import RightSidebar from './components/RightSidebar';
import TerminalFooter from './components/TerminalFooter';
import MLGalleryModal from './components/MLGalleryModal';

import initialBuildings from './data/buildings.json';
import initialRoads from './data/roads.json';
import initialWater from './data/water.json';

import { pxToLatLng } from './utils/geo';

export default function App() {
  // Global reactive state
  const [buildings, setBuildings] = useState(initialBuildings);
  const [roads] = useState(initialRoads);
  const [water] = useState(initialWater);

  const [engaged, setEngaged] = useState(false);

  // Vector Layer visibility MUST NOT auto-toggle. User manually controls checkboxes.
  const [layerVis, setLayerVis] = useState({ buildings: false, roads: false, water: false });
  const [thematicMode, setThematicMode] = useState('mat');

  // New: Triage Mode + Gallery modal state
  const [triageMode, setTriageMode] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [flyTarget, setFlyTarget] = useState(null);
  const [cursorCoords, setCursorCoords] = useState(null);

  // Toggle ENGAGE / DISENGAGE (Explicitly sets engaged state, layerVis remains under user manual control)
  const handleEngageToggle = () => {
    setEngaged(prev => {
      const next = !prev;
      if (!next) {
        setSelectedBuilding(null);
        setFlyTarget(null);
      }
      return next;
    });
  };

  // Toggle layer visibility (User manual control only)
  const handleToggleLayer = (layerName, value) => {
    setLayerVis(prev => ({ ...prev, [layerName]: value }));
  };

  // Select building polygon on map
  const handleSelectBuilding = bldg => {
    setSelectedBuilding(bldg);
  };

  // Accept building verification
  const handleAccept = id => {
    setBuildings(prev =>
      prev.map(b => (b.id === id ? { ...b, status: 'accepted' } : b))
    );
    if (selectedBuilding && selectedBuilding.id === id) {
      setSelectedBuilding(prev => ({ ...prev, status: 'accepted' }));
    }
  };

  // Reject building verification
  const handleReject = id => {
    setBuildings(prev =>
      prev.map(b => (b.id === id ? { ...b, status: 'rejected' } : b))
    );
    if (selectedBuilding && selectedBuilding.id === id) {
      setSelectedBuilding(prev => ({ ...prev, status: 'rejected' }));
    }
  };

  // Click queue card -> smooth flyToBounds to target (Does NOT alter layerVis state)
  const handleQueueRowClick = bldg => {
    setSelectedBuilding(bldg);
    if (!engaged) {
      setEngaged(true);
    }
    if (bldg.polygon_px) {
      const latlngs = bldg.polygon_px.map(p => pxToLatLng(p[0], p[1]));
      const lats = latlngs.map(ll => ll[0]);
      const lngs = latlngs.map(ll => ll[1]);
      const bounds = [
        [Math.min(...lats) - 0.0002, Math.min(...lngs) - 0.0002],
        [Math.max(...lats) + 0.0002, Math.max(...lngs) + 0.0002],
      ];
      setFlyTarget({ bounds, id: bldg.id, timestamp: Date.now() });
    }
  };

  return (
    <div className="h-screen w-screen overflow-hidden flex flex-col font-sans select-none bg-gcs-bg relative">
      {/* Scanline overlay */}
      <div className="scanline" />

      {/* ML Gallery Modal (portal-level, above everything) */}
      {galleryOpen && (
        <MLGalleryModal onClose={() => setGalleryOpen(false)} />
      )}

      {/* Top Telemetry HUD */}
      <TelemetryHUD
        buildings={buildings}
        roads={roads}
        engaged={engaged}
        onEngageToggle={handleEngageToggle}
        triageMode={triageMode}
        onOpenGallery={() => setGalleryOpen(true)}
      />

      {/* Middle Map Area */}
      <div className="flex-1 relative w-full overflow-hidden">
        <FlightMap
          buildings={buildings}
          roads={roads}
          water={water}
          layerVis={layerVis}
          thematicMode={thematicMode}
          engaged={engaged}
          flyTarget={flyTarget}
          selectedBuilding={selectedBuilding}
          onSelectBuilding={handleSelectBuilding}
          onCursorMove={setCursorCoords}
          triageMode={triageMode}
        />

        {/* Floating Left Layer Controls */}
        <LayerControls
          layerVis={layerVis}
          onToggleLayer={handleToggleLayer}
          thematicMode={thematicMode}
          onChangeThematicMode={setThematicMode}
          buildingCounts={buildings.length}
          roadCount={roads.length}
          waterCount={water.length}
          triageMode={triageMode}
          onToggleTriageMode={setTriageMode}
          onOpenGallery={() => setGalleryOpen(true)}
        />

        {/* Unified Floating Right Sidebar (Target Inspector + Human Review Queue) */}
        <RightSidebar
          buildings={buildings}
          selectedBuilding={selectedBuilding}
          onCloseInspector={() => setSelectedBuilding(null)}
          onAccept={handleAccept}
          onReject={handleReject}
          onRowClick={handleQueueRowClick}
          triageMode={triageMode}
        />
      </div>

      {/* Slim Terminal Footer */}
      <TerminalFooter cursorCoords={cursorCoords} />
    </div>
  );
}
