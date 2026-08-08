// Master Web Application Coordinator

document.addEventListener("DOMContentLoaded", function () {
  const API_BASE = "";

  // UI Elements
  const statusDot = document.getElementById("status-dot");
  const statusText = document.getElementById("status-text");

  const btnZoomIndia = document.getElementById("btn-zoom-india");
  const btnZoomCampus = document.getElementById("btn-zoom-campus");
  const btnDrawPolygon = document.getElementById("btn-draw-polygon");

  const drawnAreaStatus = document.getElementById("drawn-area-status");

  const selectModel = document.getElementById("select-model");
  const modelAccBadge = document.getElementById("model-acc-badge");
  const modelInfoBox = document.getElementById("model-info-box");

  const inputStartDate = document.getElementById("input-start-date");
  const inputEndDate = document.getElementById("input-end-date");

  const rangeCloud = document.getElementById("range-cloud");
  const cloudVal = document.getElementById("cloud-val");

  const checkSmoothing = document.getElementById("check-smoothing");
  const btnRunClassify = document.getElementById("btn-run-classify");

  const layerControls = document.getElementById("layer-controls");
  const rangeOpacityTerrain = document.getElementById("range-opacity-terrain");
  const opacityTerrainVal = document.getElementById("opacity-terrain-val");
  const rangeOpacityRgb = document.getElementById("range-opacity-rgb");
  const opacityRgbVal = document.getElementById("opacity-rgb-val");

  const analyticsPanel = document.getElementById("analytics-panel");
  const panelHeaderToggle = document.getElementById("panel-header-toggle");
  const btnTogglePanel = document.getElementById("btn-toggle-panel");

  const statModelName = document.getElementById("stat-model-name");

  const btnExportGeoTIFF = document.getElementById("btn-export-geotiff");
  const btnExportGeoJSON = document.getElementById("btn-export-geojson");

  let currentResults = null;
  let modelsMetaData = {};

  // 1. Initialize Map
  MapController.initMap("map", onShapeDrawn);

  // 2. Fetch Health & Model Metadata
  fetchHealthAndModels();

  // Event Listeners
  btnZoomIndia.addEventListener("click", () => MapController.zoomToIndia());
  btnZoomCampus.addEventListener("click", () => MapController.zoomToCampus());

  btnDrawPolygon.addEventListener("click", () => MapController.startDrawingPolygon());

  // GeoJSON upload as an alternative to drawing the area
  const btnUploadGeoJSON = document.getElementById("btn-upload-geojson");
  const inputGeoJSONFile = document.getElementById("input-geojson-file");
  btnUploadGeoJSON.addEventListener("click", () => inputGeoJSONFile.click());
  inputGeoJSONFile.addEventListener("change", async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      const geojson = JSON.parse(await file.text());
      if (!MapController.loadGeoJSON(geojson)) {
        alert("No Polygon or MultiPolygon geometry found in that GeoJSON file.");
      }
    } catch (err) {
      alert(`Could not read GeoJSON file: ${err.message}`);
    } finally {
      e.target.value = "";
    }
  });

  rangeCloud.addEventListener("input", (e) => {
    cloudVal.textContent = `${e.target.value}%`;
  });

  selectModel.addEventListener("change", (e) => {
    updateModelInfo(e.target.value);
  });

  btnRunClassify.addEventListener("click", runClassification);

  // Opacity Sliders
  rangeOpacityTerrain.addEventListener("input", (e) => {
    const opacity = parseFloat(e.target.value) / 100.0;
    opacityTerrainVal.textContent = `${e.target.value}%`;
    MapController.setTerrainOpacity(opacity);
  });

  rangeOpacityRgb.addEventListener("input", (e) => {
    const opacity = parseFloat(e.target.value) / 100.0;
    opacityRgbVal.textContent = `${e.target.value}%`;
    MapController.setRGBOpacity(opacity);
  });

  // Toggle Analytics Drawer
  panelHeaderToggle.addEventListener("click", () => {
    analyticsPanel.classList.toggle("collapsed");
  });

  // Exports
  btnExportGeoTIFF.addEventListener("click", handleExportGeoTIFF);
  btnExportGeoJSON.addEventListener("click", handleExportGeoJSON);

  // Callback when a shape is drawn on map
  function onShapeDrawn(geometry) {
    if (geometry) {
      drawnAreaStatus.style.display = "flex";
      btnRunClassify.disabled = false;
      btnRunClassify.innerHTML = `
        <svg class="icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        <span>Classify Marked Region</span>
      `;
    } else {
      drawnAreaStatus.style.display = "none";
      btnRunClassify.disabled = true;
    }
  }

  async function fetchHealthAndModels() {
    try {
      const [modelsRes, configRes] = await Promise.all([
        fetch(`${API_BASE}/api/models`),
        fetch(`${API_BASE}/api/config`)
      ]);
      if (modelsRes.ok && configRes.ok) {
        const modelsData = await modelsRes.json();
        const configData = await configRes.json();
        modelsMetaData = modelsData.models || {};
        const defaultSeason = configData.seasons?.[configData.default_season];
        if (defaultSeason) {
          inputStartDate.value = defaultSeason.start;
          inputEndDate.value = defaultSeason.end;
        }
        updateModelInfo(selectModel.value);

        statusDot.classList.add("active");
        statusText.textContent = "Earth Engine Connected";
      } else {
        statusText.textContent = "API Service Offline";
      }
    } catch (err) {
      console.warn("API health check notice:", err);
      statusText.textContent = "Connecting Earth Engine...";
    }
  }

  function updateModelInfo(modelKey) {
    const meta = modelsMetaData[modelKey];
    if (meta) {
      modelAccBadge.textContent = `Val Acc: ${meta.internal_accuracy}%`;
      modelInfoBox.innerHTML = `<strong>${meta.name} (${meta.type})</strong><br/>${meta.description}`;
    }
  }

  async function runClassification() {
    const geometry = MapController.getCurrentGeoJSON();
    if (!geometry) {
      alert("Please draw an area polygon or rectangle on the map first!");
      return;
    }

    const payload = {
      geometry: geometry,
      model_type: selectModel.value,
      start_date: inputStartDate.value,
      end_date: inputEndDate.value,
      cloud_threshold: parseFloat(rangeCloud.value),
      smoothing: checkSmoothing.checked
    };

    btnRunClassify.disabled = true;
    btnRunClassify.innerHTML = `<span class="spinner"></span> <span>Processing GEE Inference...</span>`;

    try {
      const response = await fetch(`${API_BASE}/api/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Classification failed");
      }

      const results = await response.json();
      currentResults = results;

      // Update map overlays (single pre-rendered images, not live GEE tiles)
      MapController.updateOverlays(results.rgb_overlay, results.terrain_overlay);

      // Show Layer Controls
      layerControls.style.display = "flex";

      // Render Analytics & Individual Class Areas
      statModelName.textContent = results.model.name;
      ChartsController.renderTerrainAnalytics(results.summary, results.individual_class_areas);

      // Expand Analytics Panel
      analyticsPanel.style.display = "flex";
      analyticsPanel.classList.remove("collapsed");

    } catch (err) {
      alert(`Classification Error: ${err.message}`);
      console.error(err);
    } finally {
      btnRunClassify.disabled = false;
      btnRunClassify.innerHTML = `
        <svg class="icon-md" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"/>
        </svg>
        <span>Re-Classify Region</span>
      `;
    }
  }

  function handleExportGeoTIFF() {
    if (!currentResults || !currentResults.export.geotiff_url) {
      alert("No active GeoTIFF export URL available. Please run classification first!");
      return;
    }
    window.open(currentResults.export.geotiff_url, "_blank");
  }

  function handleExportGeoJSON() {
    if (!currentResults) {
      alert("No classification results to export. Please run classification first!");
      return;
    }

    const exportData = {
      type: "Feature",
      geometry: MapController.getCurrentGeoJSON(),
      properties: {
        model: currentResults.model,
        summary: currentResults.summary,
        individual_class_areas: currentResults.individual_class_areas
      }
    };

    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
    const downloadAnchor = document.createElement("a");
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `terrain_report_${currentResults.model.id}.geojson`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }
});
