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
import UploadModal from './components/UploadModal';

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
  // ── Core state (must be defined before any derived memo/effect) ──────
  const [selectedVillageId, setSelectedVillageId] = useState(DEFAULT_VILLAGE_ID);
  const [pickerOpen, setPickerOpen] = useState(true);
  const [pickerDismissible, setPickerDismissible] = useState(true);
  const [customUpload, setCustomUpload] = useState(null); // {imageDataUrl, buildings, roads, water, width, height, counts, gsd}
  const [uploadOpen, setUploadOpen] = useState(false);
  const [engaged, setEngaged] = useState(false);
  const [layerVis, setLayerVis] = useState({
    buildings: false,
    roads: false,
    water: true,
  });
  const [thematicMode, setThematicMode] = useState('mat');
  const [triageMode, setTriageMode] = useState(false);
  const [galleryOpen, setGalleryOpen] = useState(false);
  const [metricsOpen, setMetricsOpen] = useState(false);
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [flyTarget, setFlyTarget] = useState(null);
  const [cursorCoords, setCursorCoords] = useState(null);
  const [appStatus, setAppStatus] = useState(APP_STATUS.PROCESSING);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [systemStatusCollapsed, setSystemStatusCollapsed] = useState(false);
  const [exportNotice, setExportNotice] = useState('');

  const activeVillage = useMemo(() => customUpload ? {
    id: 'custom_upload',
    name: `Uploaded · ${customUpload.width}×${customUpload.height} · ${customUpload.gsd} m/px`,
    image: customUpload.imageDataUrl,
    bbox: { x0: 0, x1: customUpload.width, y0: 0, y1: customUpload.height },
    featureAvailability: { buildings: true, roads: true, water: true },
  } : getVillageOption(selectedVillageId), [customUpload, selectedVillageId]);

  // Re-filter the SAME real dataset down to whichever photo/region is active.
  const filteredInitialData = useMemo(
    () => customUpload ? { buildings: customUpload.buildings, roads: customUpload.roads, water: customUpload.water }
      : filterAllByRegion(initialBuildingsRaw, initialRoadsRaw, initialWaterRaw, activeVillage.bbox),
    [activeVillage, customUpload]
  );

  // Global reactive state — init from filtered data on first mount
  const initialFeatures = activeVillage.featureAvailability || { buildings: true, roads: true, water: true };
  const [buildings, setBuildings] = useState(() => initialFeatures.buildings ? filteredInitialData.buildings : []);
  const [roads, setRoads] = useState(() => initialFeatures.roads ? filteredInitialData.roads : []);
  const [water, setWater] = useState(() => initialFeatures.water ? filteredInitialData.water : []);

  // Sync when switching village or after upload (but not on first mount to avoid flash)
  const isFirstSync = React.useRef(true);
  useEffect(() => {
    if (isFirstSync.current) { isFirstSync.current = false; return; }
    setBuildings(initialFeatures.buildings ? filteredInitialData.buildings : []);
    setRoads(initialFeatures.roads ? filteredInitialData.roads : []);
    setWater(initialFeatures.water ? filteredInitialData.water : []);
    if (customUpload) {
      setLayerVis(prev => ({ buildings: true, roads: false, water: true })); // roads OFF by default per request
      setEngaged(true);
      setFlyTarget({ bounds: bboxToLatLngBounds(activeVillage.bbox), id: 'custom-upload', timestamp: Date.now() });
      setCurrentStageIndex(0); setAppStatus(APP_STATUS.PROCESSING);
    }
  }, [filteredInitialData]);

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
  const handleUploadComplete = data => {
    setCustomUpload(data);
    setSelectedBuilding(null);
    setPickerOpen(false);
  };
  const handleClearCustom = () => {
    setCustomUpload(null);
    setSelectedBuilding(null);
    handleSelectVillage(DEFAULT_VILLAGE_ID);
  };

  const handleSelectVillage = id => {
    if (customUpload && id !== 'custom_upload') setCustomUpload(null);
    const village = getVillageOption(id);
    const filtered = filterAllByRegion(initialBuildingsRaw, initialRoadsRaw, initialWaterRaw, village.bbox);

    const features = village.featureAvailability || { buildings: true, roads: true, water: true };
    setSelectedVillageId(id);
    setBuildings(features.buildings ? filtered.buildings : []);
    setRoads(features.roads ? filtered.roads : []);
    setWater(features.water ? filtered.water : []);
    setLayerVis(prev => ({
      buildings: features.buildings && filtered.buildings.length > 0,
      roads: false, // OFF by default — user must toggle ROAD layer on per request
      water: features.water && filtered.water.length > 0,
    }));

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

      {exportNotice && (
        <div className="export-toast">
          <span className="export-toast-dot" />
          {exportNotice}
        </div>
      )}

      {/* Demo Region / Village Picker (portal-level, above everything) */}
      {pickerOpen && (
        <VillagePicker
          activeId={customUpload ? 'custom_upload' : selectedVillageId}
          dismissible={pickerDismissible}
          onSelect={handleSelectVillage}
          onClose={() => setPickerOpen(false)}
        />
      )}

      {/* Upload Modal */}
      {uploadOpen && (
        <UploadModal onClose={() => setUploadOpen(false)} onUploadComplete={handleUploadComplete} />
      )}
      {customUpload && (
        <div className="fixed top-[48px] left-1/2 -translate-x-1/2 z-[2000] bg-gcs-cyan text-black font-mono text-xs px-3 py-1.5 rounded flex items-center gap-3 shadow-lg">
          <span>⬆ Uploaded: {customUpload.width}×{customUpload.height} · {customUpload.counts.buildings} bldgs · {customUpload.counts.roads} roads · {customUpload.counts.total_road_m} m · {customUpload.elapsed}s</span>
          <button onClick={handleClearCustom} className="bg-black text-gcs-cyan px-2 py-0.5 rounded text-[11px] hover:bg-slate-800">✕ Clear</button>
        </div>
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
        onOpenVillagePicker={() => setPickerOpen(true)}
        onOpenUpload={() => setUploadOpen(true)}
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

        {/* Left Side Panels (Stacked) */}
        <div className="absolute left-4 top-24 bottom-12 w-[220px] flex flex-col gap-3 z-[1000] pointer-events-none">
          {/* Floating Left Layer Controls */}
          <div className="pointer-events-auto flex-1 min-h-0 flex flex-col">
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
          </div>

          {/* Floating System Status Panel */}
          <div className="pointer-events-auto shrink-0">
            <SystemStatus
              status={systemStatus}
              collapsed={systemStatusCollapsed}
              onToggleCollapsed={() => setSystemStatusCollapsed(prev => !prev)}
            />
          </div>
        </div>

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
      <TerminalFooter
        cursorCoords={cursorCoords}
        onOpenGallery={() => setGalleryOpen(true)}
        onOpenMetrics={() => setMetricsOpen(true)}
        onExportCsv={handleExportCsv}
        buildings={buildings}
        triageMode={triageMode}
      />
    </div>
  );
}
