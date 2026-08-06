import { LngLatBounds, Map as MLMap, NavigationControl, ScaleControl, prewarm } from "maplibre-gl";
import type { TerraDraw } from "terra-draw";
import type { Polygon } from "geojson";
import type { OverlayInfo } from "./api";

export const INDIA = { center: [78.9629, 20.5937] as [number, number], zoom: 4 };
export const CAMPUS = { center: [80.026, 23.174] as [number, number], zoom: 14 };

// Sentinel-2 is 10 m/px. Web Mercator resolution is 156543*cos(lat)/2^z, so at
// ~23N a 256px tile hits 10 m/px around zoom 14. Every zoom past that is pure
// magnification of pixels that do not exist -- and against a live Earth Engine
// endpoint each of those empty tiles still costs a full inference. Capping the
// source here lets the GPU upscale instead, which is free.
const S2_NATIVE_MAXZOOM = 14;

let map: MLMap;
let aoi: Polygon | null = null;
let rgbTiles: string | null = null;
let rgbAdded = false;

let drawing: Promise<TerraDraw> | null = null;
let onAoiChange: (g: Polygon | null) => void = () => {};

/** The drawing engine is ~80 KB and nobody can draw before the map has painted,
 *  so it is fetched on the first click of a draw button, not on page load. */
function getDraw(): Promise<TerraDraw> {
  drawing ??= (async () => {
    const [td, adapter] = await Promise.all([
      import("terra-draw"),
      import("terra-draw-maplibre-gl-adapter"),
    ]);
    const draw = new td.TerraDraw({
      adapter: new adapter.TerraDrawMapLibreGLAdapter({ map }),
      // terra-draw's rectangle mode defaults to drawInteraction "click-move",
      // which makes its onDragStart/onDrag/onDragEnd no-ops: dragging draws
      // nothing and the AOI never completes. "click-move-or-drag" keeps the
      // click-two-corners gesture and re-enables the natural drag-to-draw.
      modes: [
        new td.TerraDrawRectangleMode({ drawInteraction: "click-move-or-drag" }),
        new td.TerraDrawPolygonMode(),
      ],
    });
    draw.start();

    draw.on("finish", (id) => {
      const feature = draw.getSnapshotFeature(id);
      if (!feature || feature.geometry.type !== "Polygon") return;
      // One AOI at a time -- drop anything drawn earlier.
      for (const f of draw.getSnapshot()) if (f.id !== id) draw.removeFeatures([f.id!]);
      aoi = feature.geometry as Polygon;
      draw.setMode("static");
      onAoiChange(aoi);
    });
    return draw;
  })();
  return drawing;
}

export function initMap(container: string, onAoi: (g: Polygon | null) => void): MLMap {
  onAoiChange = onAoi;
  // Spin the render workers up before the style lands so first paint isn't
  // waiting on worker boot.
  prewarm();

  map = new MLMap({
    container,
    style: {
      version: 8,
      sources: {
        esri: {
          type: "raster",
          tiles: ["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"],
          tileSize: 256,
          maxzoom: 19,
          attribution: "Esri World Imagery",
        },
      },
      layers: [{ id: "esri", type: "raster", source: "esri" }],
    },
    center: INDIA.center,
    zoom: INDIA.zoom,
    attributionControl: { compact: true },
  });

  map.on("error", (e) => console.error("[map]", e.error?.message ?? e));

  map.addControl(new NavigationControl({ showCompass: false }), "top-right");
  map.addControl(new ScaleControl({ unit: "metric" }), "bottom-left");

  return map;
}

/** Drop the classified terrain/rgb overlays.
 *
 *  They are added above every other layer, and the terrain one is opaque, so a
 *  rectangle drawn after a classification would sit underneath it and be
 *  invisible. Starting a new draw or clearing must remove them first. */
export function clearResultOverlays() {
  removeLayer("terrain");
  removeLayer("rgb");
  rgbAdded = false;
}

export const setDrawMode = async (mode: "rectangle" | "polygon") => {
  // A previous classification's opaque overlay would hide the new selection.
  clearResultOverlays();
  (await getDraw()).setMode(mode);
};

export async function clearAoi() {
  const draw = await getDraw();
  draw.clear();
  draw.setMode("static");
  aoi = null;
  clearResultOverlays();
  onAoiChange(null);
}

export const getAoi = () => aoi;

export const getMap = () => map;

export const flyTo = (to: { center: [number, number]; zoom: number }) =>
  map.flyTo({ ...to, duration: 1200 });

function removeLayer(id: string) {
  if (map.getLayer(id)) map.removeLayer(id);
  if (map.getSource(id)) map.removeSource(id);
}

/**
 * Drop the classified result on the map as one flat image.
 *
 * `raster-resampling: nearest` matters for correctness, not just crispness:
 * these are class IDs painted as colours, so bilinear filtering would blend
 * Forest into Water and render a class that was never predicted.
 */
export function showTerrain(overlay: OverlayInfo, opacity = 1, fallbackTiles?: string): Promise<void> {
  removeLayer("terrain");

  // The backend returns an empty url if Earth Engine refused to render the
  // thumbnail. Fall back to the live tile endpoint -- slow, but a slow map
  // beats no map.
  if (!overlay.url) {
    if (!fallbackTiles) return Promise.resolve();
    map.addSource("terrain", { type: "raster", tiles: [fallbackTiles], tileSize: 256, maxzoom: S2_NATIVE_MAXZOOM });
    map.addLayer({ id: "terrain", type: "raster", source: "terrain", paint: { "raster-opacity": opacity } });
    return terrainLoaded();
  }

  const { west, south, east, north } = overlay.bounds;
  map.addSource("terrain", {
    type: "image",
    url: overlay.url,
    coordinates: [[west, north], [east, north], [east, south], [west, south]],
  });
  map.addLayer({
    id: "terrain",
    type: "raster",
    source: "terrain",
    paint: { "raster-opacity": opacity, "raster-resampling": "nearest", "raster-fade-duration": 0 },
  });
  return terrainLoaded();
}

/** Resolves once the terrain raster is actually on screen.
 *
 *  Earth Engine renders the overlay on demand, so this can take double-digit
 *  seconds; the caller keeps its progress readout up until then rather than
 *  declaring victory over a still-empty map. Always settles: a stalled render
 *  must not leave the UI stuck mid-run. */
function terrainLoaded(timeoutMs = 90000): Promise<void> {
  return new Promise((resolve) => {
    const finish = () => {
      map.off("sourcedata", onData);
      map.off("error", finish);
      clearTimeout(timer);
      resolve();
    };
    const onData = (e: { sourceId?: string; isSourceLoaded?: boolean }) => {
      if (e.sourceId === "terrain" && e.isSourceLoaded) finish();
    };
    const timer = setTimeout(finish, timeoutMs);
    map.on("sourcedata", onData);
    map.on("error", finish);
  });
}

/** Remembered, not loaded: RGB tiles are live inference and sit under an opaque
 *  terrain layer, so fetching them before they can be seen is wasted work. */
export const setRgbSource = (tiles: string) => {
  // Drop any layer left over from a previous run, or re-adding the source on
  // the next slider move throws.
  removeLayer("rgb");
  rgbTiles = tiles;
  rgbAdded = false;
};

export function setOpacity(layer: "terrain" | "rgb", value: number) {
  if (layer === "rgb") {
    if (value > 0 && !rgbAdded && rgbTiles) {
      map.addSource("rgb", { type: "raster", tiles: [rgbTiles], tileSize: 256, maxzoom: S2_NATIVE_MAXZOOM });
      // Under the classified layer when there is one; addLayer rejects a
      // beforeId that does not exist.
      map.addLayer(
        { id: "rgb", type: "raster", source: "rgb", paint: { "raster-opacity": value } },
        map.getLayer("terrain") ? "terrain" : undefined,
      );
      rgbAdded = true;
      return;
    }
    if (!rgbAdded) return;
  }
  if (map.getLayer(layer)) map.setPaintProperty(layer, "raster-opacity", value);
}

export function fitToAoi() {
  if (!aoi) return;
  const b = new LngLatBounds();
  for (const [lng, lat] of aoi.coordinates[0]) b.extend([lng, lat]);
  map.fitBounds(b, { padding: 60, maxZoom: 16, duration: 600 });
}

/** Rough AOI size for the sidebar readout. */
export function aoiExtentKm(g: Polygon): string {
  const lngs = g.coordinates[0].map((c) => c[0]);
  const lats = g.coordinates[0].map((c) => c[1]);
  const mid = ((Math.min(...lats) + Math.max(...lats)) / 2) * (Math.PI / 180);
  const w = (Math.max(...lngs) - Math.min(...lngs)) * 111.32 * Math.cos(mid);
  const h = (Math.max(...lats) - Math.min(...lats)) * 110.54;
  return `${w.toFixed(1)} × ${h.toFixed(1)} km`;
}
