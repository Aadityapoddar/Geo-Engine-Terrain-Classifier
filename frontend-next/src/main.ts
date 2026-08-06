import "maplibre-gl/dist/maplibre-gl.css";
import "./style.css";
import type { Polygon } from "geojson";
import { classify, getConfig, getModels, type ClassifyResult, type ModelMeta } from "./api";
import { renderClassCards, renderDonut } from "./donut";
import * as M from "./map";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const el = {
  statusDot: $("status-dot"),
  statusText: $("status-text"),
  drawRect: $<HTMLButtonElement>("btn-draw-rect"),
  drawPoly: $<HTMLButtonElement>("btn-draw-poly"),
  drawClear: $<HTMLButtonElement>("btn-draw-clear"),
  aoiStatus: $("drawn-area-status"),
  aoiSize: $("aoi-size"),
  model: $<HTMLSelectElement>("select-model"),
  modelBadge: $("model-acc-badge"),
  modelInfo: $("model-info-box"),
  startDate: $<HTMLInputElement>("input-start-date"),
  endDate: $<HTMLInputElement>("input-end-date"),
  cloud: $<HTMLInputElement>("range-cloud"),
  cloudVal: $("cloud-val"),
  smoothing: $<HTMLInputElement>("check-smoothing"),
  run: $<HTMLButtonElement>("btn-run-classify"),
  progress: $("run-progress"),
  progressFill: $("run-progress-fill"),
  progressLabel: $("run-progress-label"),
  layerControls: $("layer-controls"),
  opacityTerrain: $<HTMLInputElement>("range-opacity-terrain"),
  opacityTerrainVal: $("opacity-terrain-val"),
  opacityRgb: $<HTMLInputElement>("range-opacity-rgb"),
  opacityRgbVal: $("opacity-rgb-val"),
  panel: $("analytics-panel"),
  panelToggle: $("panel-header-toggle"),
  statTotal: $("stat-total-ha"),
  statDominant: $("stat-dominant"),
  statModel: $("stat-model-name"),
  statTime: $("stat-proc-time"),
  donut: $("donut-wrapper"),
  classList: $("class-areas-list"),
  exportTiff: $<HTMLButtonElement>("btn-export-geotiff"),
  exportPng: $<HTMLButtonElement>("btn-export-png"),
  exportJson: $<HTMLButtonElement>("btn-export-geojson"),
  searchForm: $<HTMLFormElement>("search-form"),
  searchInput: $<HTMLInputElement>("search-input"),
};

let models: Record<string, ModelMeta> = {};
let result: ClassifyResult | null = null;

M.initMap("map", onAoi);

// Handle for the benchmark harness, so it drives the real map code path rather
// than a reimplementation of it. Absent from a plain production build.
if (import.meta.env.DEV || import.meta.env.VITE_BENCH)
  (window as unknown as Record<string, unknown>).__geo = M;

Promise.all([getModels(), getConfig()])
  .then(([modelData, config]) => {
    models = modelData.models ?? {};
    const season = config.seasons[config.default_season];
    if (season) {
      el.startDate.value = season.start;
      el.endDate.value = season.end;
    }
    showModel(el.model.value);
    el.statusDot.classList.add("active");
    el.statusText.textContent = "Earth Engine Connected";
  })
  .catch(() => {
    el.statusText.textContent = "API Service Offline";
  });

function showModel(key: string) {
  const m = models[key];
  if (!m) return;
  el.modelBadge.textContent = `Val Acc: ${m.internal_accuracy}%`;
  el.modelInfo.innerHTML = `<strong>${m.name} (${m.type})</strong><br/>${m.description}`;
}

function onAoi(g: Polygon | null) {
  el.aoiStatus.hidden = !g;
  el.run.disabled = !g;
  if (g) el.aoiSize.textContent = M.aoiExtentKm(g);
}

el.drawRect.addEventListener("click", () => M.setDrawMode("rectangle"));
el.drawPoly.addEventListener("click", () => M.setDrawMode("polygon"));
el.drawClear.addEventListener("click", () => {
  M.clearAoi();
  el.layerControls.hidden = true;
  el.panel.hidden = true;
});

$("btn-zoom-india").addEventListener("click", () => M.flyTo(M.INDIA));
$("btn-zoom-campus").addEventListener("click", () => M.flyTo(M.CAMPUS));

el.model.addEventListener("change", () => showModel(el.model.value));
el.cloud.addEventListener("input", () => (el.cloudVal.textContent = `${el.cloud.value}%`));

el.opacityTerrain.addEventListener("input", () => {
  el.opacityTerrainVal.textContent = `${el.opacityTerrain.value}%`;
  M.setOpacity("terrain", +el.opacityTerrain.value / 100);
});
el.opacityRgb.addEventListener("input", () => {
  el.opacityRgbVal.textContent = `${el.opacityRgb.value}%`;
  M.setOpacity("rgb", +el.opacityRgb.value / 100);
});

el.panelToggle.addEventListener("click", () => el.panel.classList.toggle("collapsed"));

/** The request takes tens of seconds; a spinner with no elapsed time reads as a hang. */
function startProgress() {
  const t0 = performance.now();
  el.progress.hidden = false;
  let override: string | null = null;
  const stages: [number, string][] = [
    [0, "Training classifier on labelled points…"],
    [8, "Building Sentinel-2 composite…"],
    [20, "Running per-pixel inference…"],
  ];
  const tick = setInterval(() => {
    const s = (performance.now() - t0) / 1000;
    const stage = override ?? [...stages].reverse().find(([at]) => s >= at)![1];
    el.progressLabel.textContent = `${stage}  ${s.toFixed(0)}s`;
  }, 200);
  return {
    setStage: (s: string) => (override = s),
    stop: () => {
      clearInterval(tick);
      el.progress.hidden = true;
    },
  };
}

el.run.addEventListener("click", async () => {
  const geometry = M.getAoi();
  if (!geometry) return;

  el.run.disabled = true;
  el.run.textContent = "Processing…";
  const progress = startProgress();

  try {
    result = await classify({
      geometry,
      model_type: el.model.value,
      start_date: el.startDate.value,
      end_date: el.endDate.value,
      cloud_threshold: +el.cloud.value,
      smoothing: el.smoothing.checked,
    });

    M.setRgbSource(result.tile_urls.sentinel_rgb);
    // Earth Engine renders the overlay when it is requested, so the numbers land
    // well before the raster does. Show them straight away, keep the progress
    // readout up until the map itself is populated.
    const terrainReady = M.showTerrain(
      result.terrain_overlay,
      +el.opacityTerrain.value / 100,
      result.tile_urls.terrain_classified,
    );
    progress.setStage("Rendering classified raster…");
    M.fitToAoi();

    el.layerControls.hidden = false;
    el.opacityRgb.value = "0";
    el.opacityRgbVal.textContent = "0%";

    const classes = result.individual_class_areas;
    const dominant = classes.reduce((a, b) => (b.percentage > a.percentage ? b : a));
    el.statTotal.textContent = `${result.summary.total_area_ha.toLocaleString()} ha`;
    el.statDominant.textContent = `${dominant.name} (${dominant.percentage}%)`;
    el.statModel.textContent = result.model.name;
    el.statTime.textContent = `${result.summary.processing_time_sec}s`;
    renderDonut(el.donut, classes);
    renderClassCards(el.classList, classes);

    el.panel.hidden = false;
    el.panel.classList.remove("collapsed");

    await terrainReady;
  } catch (err) {
    el.statusText.textContent = `Classification failed: ${(err as Error).message}`;
  } finally {
    progress.stop();
    el.run.disabled = false;
    el.run.textContent = "Re-Classify Region";
  }
});

el.exportTiff.addEventListener("click", () => {
  if (result?.export.geotiff_url) window.open(result.export.geotiff_url, "_blank");
});
el.exportPng.addEventListener("click", () => {
  if (result?.terrain_overlay.url) window.open(result.terrain_overlay.url, "_blank");
});
el.exportJson.addEventListener("click", () => {
  if (!result) return;
  const blob = new Blob(
    [
      JSON.stringify(
        {
          type: "Feature",
          geometry: M.getAoi(),
          properties: {
            model: result.model,
            summary: result.summary,
            individual_class_areas: result.individual_class_areas,
          },
        },
        null,
        2,
      ),
    ],
    { type: "application/geo+json" },
  );
  const a = Object.assign(document.createElement("a"), {
    href: URL.createObjectURL(blob),
    download: `terrain_report_${result.model.id}.geojson`,
  });
  a.click();
  URL.revokeObjectURL(a.href);
});

// ponytail: 15 lines against Nominatim replaces an 11 KB geocoder plugin.
el.searchForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = el.searchInput.value.trim();
  if (!q) return;
  const res = await fetch(
    `https://nominatim.openstreetmap.org/search?format=json&limit=1&q=${encodeURIComponent(q)}`,
  ).then((r) => r.json());
  if (!res[0]) {
    el.searchInput.setAttribute("placeholder", "No match - try again");
    return;
  }
  M.flyTo({ center: [+res[0].lon, +res[0].lat], zoom: 13 });
});
