import React, { useState, useEffect, useMemo } from 'react';
import TelemetryHUD from './components/TelemetryHUD';
import FlightMap from './components/FlightMap';
import LayerControls from './components/LayerControls';
import RightSidebar from './components/RightSidebar';
import TerminalFooter from './components/TerminalFooter';
import MLGalleryModal from './components/MLGalleryModal';
import SystemStatus from './components/SystemStatus';
import MetricsPage from './components/MetricsPage';
import VillagePicker from './components/VillagePicker';

import initialBuildingsRaw from './data/buildings.json';
import initialRoadsRaw from './data/roads.json';
import initialWaterRaw from './data/water.json';

import { pxToLatLng, bboxToLatLngBounds } from './utils/geo';
import { APP_STATUS, PROCESSING_STAGES, computeSystemStatus } from './utils/systemStatus';
import { getVillageOption } from './data/villageOptions';
import { filterAllByRegion } from './utils/regionFilter';
import { exportBuildingResultsCsv } from './utils/exportCsv';

// How long each simulated processing stage takes to "complete" on load,
// so the System Status layer has something real to narrate. This mirrors
// the actual pipeline order (imagery -> buildings -> roads/water -> confidence
// -> review priority) even though the JSON is pre-computed for the demo.
const STAGE_DURATION_MS = 380;

const DEFAULT_VILLAGE_ID = 'area_01';

export default function App() {
  // ── Region / Village selection (the "pick between photos" feature) ──────
  const [selectedVillageId, setSelectedVillageId] = useState(DEFAULT_VILLAGE_ID);
  const [pickerOpen, setPickerOpen] = useState(true); // shown on first load
  const [pickerDismissible, setPickerDismissible] = useState(true); // 'full' is preselected, so closing without picking is fine

  const activeVillage = getVillageOption(selectedVillageId);

  // Re-filter the SAME real dataset down to whichever photo/region is active.
  const filteredInitialData = useMemo(
    () => filterAllByRegion(initialBuildingsRaw, initialRoadsRaw, initialWaterRaw, activeVillage.bbox),
    [activeVillage]
  );

  // Global reactive state
  const initialFeatures = activeVillage.featureAvailability || { buildings: true, roads: true, water: true };
  const [buildings, setBuildings] = useState(initialFeatures.buildings ? filteredInitialData.buildings : []);
  const [roads, setRoads] = useState(initialFeatures.roads ? filteredInitialData.roads : []);
  const [water, setWater] = useState(initialFeatures.water ? filteredInitialData.water : []);

  const [engaged, setEngaged] = useState(false);

  // Vector Layer visibility is user-controlled; switching survey areas initializes sensible defaults.
  const [layerVis, setLayerVis] = useState({ buildings: true, roads: true, water: false });
  const [thematicMode, setThematicMode] = useState('mat');

  // New: Triage Mode + Gallery modal state
  const [triageMode, setTriageMode] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);

  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [flyTarget, setFlyTarget] = useState(null);
  const [cursorCoords, setCursorCoords] = useState(null);

  // ── System Status: PROCESSING -> OPERATIONAL pipeline simulation ─────────
  const [appStatus, setAppStatus] = useState(APP_STATUS.PROCESSING);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [systemStatusCollapsed, setSystemStatusCollapsed] = useState(false);
  const [exportNotice, setExportNotice] = useState('');

  useEffect(() => {
    if (appStatus !== APP_STATUS.PROCESSING) return;

    if (currentStageIndex >= PROCESSING_STAGES.length - 1) {
      const finishTimer = setTimeout(() => setAppStatus(APP_STATUS.OPERATIONAL), STAGE_DURATION_MS);
      return () => clearTimeout(finishTimer);
    }

    const stepTimer = setTimeout(() => setCurrentStageIndex(i => i + 1), STAGE_DURATION_MS);
    return () => clearTimeout(stepTimer);
  }, [appStatus, currentStageIndex]);

  const systemStatus = computeSystemStatus(buildings, roads, water, {
    appStatus,
    currentStageIndex,
    datasetName: activeVillage.name,
    mode: 'DEMO',
    processingTimeSec: 6.8,
  });

  // Load a different demo survey area: re-filters the prototype dataset, resets
  // review state for that subset, and replays the PROCESSING -> OPERATIONAL
  // pipeline so System Status narrates a genuine reload.
  const handleSelectVillage = id => {
    const village = getVillageOption(id);
    const filtered = filterAllByRegion(initialBuildingsRaw, initialRoadsRaw, initialWaterRaw, village.bbox);

    const features = village.featureAvailability || { buildings: true, roads: true, water: true };
    setSelectedVillageId(id);
    setBuildings(features.buildings ? filtered.buildings : []);
    setRoads(features.roads ? filtered.roads : []);
    setWater(features.water ? filtered.water : []);
    setLayerVis({
      buildings: features.buildings && filtered.buildings.length > 0,
      roads: features.roads && filtered.roads.length > 0,
      water: features.water && filtered.water.length > 0,
    });

    setSelectedBuilding(null);
    setPickerOpen(false);

    // Replay the load pipeline for the newly selected region.
    setCurrentStageIndex(0);
    setAppStatus(APP_STATUS.PROCESSING);

    // Fly the map to the new region so the switch is visible, not just a stat change.
    setEngaged(true);
    setFlyTarget({ bounds: bboxToLatLngBounds(village.bbox), id: `region-${id}`, timestamp: Date.now() });
  };

  const handleExportCsv = () => {
    if (!buildings.length) return;
    const result = exportBuildingResultsCsv(buildings, activeVillage);
    setExportNotice(`${result.count} records exported`);
    window.setTimeout(() => setExportNotice(''), 2400);
  };

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
      const { x0, x1, y0, y1 } = activeVillage.bbox;
      const latlngs = bldg.polygon_px.map(p => {
        const x = ((p[0] - x0) / (x1 - x0)) * 1000;
        const y = ((p[1] - y0) / (y1 - y0)) * 1000;
        return pxToLatLng(x, y);
      });
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

      {exportNotice && (
        <div className="export-toast">
          <span className="export-toast-dot" />
          {exportNotice}
        </div>
      )}

      {/* Demo Region / Village Picker (portal-level, above everything) */}
      {pickerOpen && (
        <VillagePicker
          activeId={selectedVillageId}
          dismissible={pickerDismissible}
          onSelect={handleSelectVillage}
          onClose={() => setPickerOpen(false)}
        />
      )}

      {/* ML Gallery Modal (portal-level, above everything) */}
      {galleryOpen && (
        <MLGalleryModal onClose={() => setGalleryOpen(false)} />
      )}

      {/* Overall Metrics Page Modal (portal-level, above everything) */}
      {metricsOpen && (
        <MetricsPage buildings={buildings} onClose={() => setMetricsOpen(false)} />
      )}

      {/* Top Telemetry HUD */}
      <TelemetryHUD
        buildings={buildings}
        roads={roads}
        engaged={engaged}
        onEngageToggle={handleEngageToggle}
        triageMode={triageMode}
        onOpenGallery={() => setGalleryOpen(true)}
        onOpenMetrics={() => setMetricsOpen(true)}
        onOpenVillagePicker={() => setPickerOpen(true)}
        onExportCsv={handleExportCsv}
        activeVillageName={activeVillage.name}
      />

      {/* Middle Map Area */}
      <div className="flex-1 relative w-full overflow-hidden">
        <FlightMap
          buildings={buildings}
          roads={roads}
          water={water}
          activeVillage={activeVillage}
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

        {/* Floating System Status Panel (bottom-left, above the footer) */}
        <SystemStatus
          status={systemStatus}
          collapsed={systemStatusCollapsed}
          onToggleCollapsed={() => setSystemStatusCollapsed(prev => !prev)}
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
