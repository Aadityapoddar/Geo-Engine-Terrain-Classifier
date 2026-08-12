// Map Controller Module using Leaflet and Leaflet Geoman

const MapController = (function () {
  let map = null;
  let drawnItems = null;
  let currentGeoJSON = null;
  let shapeDrawnCallback = null;

  let rgbTileLayer = null;
  let terrainTileLayer = null;

  function clearOverlays() {
    if (rgbTileLayer) {
      map.removeLayer(rgbTileLayer);
      rgbTileLayer = null;
    }
    if (terrainTileLayer) {
      map.removeLayer(terrainTileLayer);
      terrainTileLayer = null;
    }
  }

  function setSelectionFill(fillOpacity) {
    if (!drawnItems) return;
    drawnItems.eachLayer((layer) => {
      if (typeof layer.setStyle === "function") {
        layer.setStyle({ fillOpacity });
      }
    });
  }

  const INDIA_CENTER = [20.5937, 78.9629];
  const INDIA_ZOOM = 5;

  const CAMPUS_CENTER = [23.174, 80.026];
  const CAMPUS_ZOOM = 15;

  function initMap(containerId, onShapeDrawnCallback) {
    shapeDrawnCallback = onShapeDrawnCallback;
    // 1. Initialize Leaflet Map over India by default
    map = L.map(containerId, {
      center: INDIA_CENTER,
      zoom: INDIA_ZOOM,
      zoomControl: false,
      attributionControl: false
    });

    // Add zoom control top right
    L.control.zoom({ position: 'topright' }).addTo(map);

    // 2. Add Basemaps (Esri Satellite & OpenStreetMap)
    const esriSatellite = L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      { maxZoom: 19, attribution: 'Esri World Imagery' }
    ).addTo(map);

    const osmBase = L.tileLayer(
      'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: '© OpenStreetMap contributors' }
    );

    const baseMaps = {
      'Esri Satellite': esriSatellite,
      'OpenStreetMap': osmBase
    };
    L.control.layers(baseMaps, null, { position: 'topright' }).addTo(map);

    // 3. Add Geocoder Search Control
    if (window.L.Control.Geocoder) {
      L.Control.geocoder({
        defaultMarkGeocode: true,
        position: 'topright',
        placeholder: 'Search location in India or world...'
      }).addTo(map);
    }

    // 4. Setup Leaflet Geoman for drawing shapes
    drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);

    map.pm.addControls({
      position: 'topleft',
      drawCircleMarker: false,
      drawPolyline: false,
      drawCircle: false,
      drawText: false,
      drawMarker: false,
      cutPolygon: false
    });

    // Customize Geoman style for drawn polygons
    map.pm.setPathOptions({
      color: '#FF002B',
      fillColor: '#FF002B',
      fillOpacity: 0.2,
      weight: 3
    });

    // Ensure map recalculates container dimensions after layout calculation
    setTimeout(function () {
      if (map) {
        map.invalidateSize();
      }
    }, 300);

    window.addEventListener('load', function () {
      if (map) {
        map.invalidateSize();
      }
    });

    // Handle Shape Draw Events
    map.on('pm:create', function (e) {
      clearOverlays();
      // Remove previous drawn shapes to maintain a single ROI
      drawnItems.clearLayers();
      
      const layer = e.layer;
      layer.setStyle({ fillOpacity: 0.2 });
      drawnItems.addLayer(layer);

      const geojson = layer.toGeoJSON();
      currentGeoJSON = geojson.geometry;

      if (onShapeDrawnCallback && typeof onShapeDrawnCallback === 'function') {
        onShapeDrawnCallback(currentGeoJSON);
      }
    });

    map.on('pm:remove', function () {
      if (drawnItems.getLayers().length === 0) {
        currentGeoJSON = null;
        if (onShapeDrawnCallback && typeof onShapeDrawnCallback === 'function') {
          onShapeDrawnCallback(null);
        }
      }
    });

    return map;
  }

  // Load an uploaded GeoJSON (Feature, FeatureCollection or bare Geometry) as
  // the marked area. Uses the first Polygon/MultiPolygon found.
  function loadGeoJSON(geojson) {
    if (!map) return false;

    let geometry = null;
    if (geojson.type === "FeatureCollection") {
      const feat = (geojson.features || []).find(
        (f) => f.geometry && /Polygon$/.test(f.geometry.type)
      );
      geometry = feat ? feat.geometry : null;
    } else if (geojson.type === "Feature") {
      geometry = geojson.geometry;
    } else if (/Polygon$/.test(geojson.type)) {
      geometry = geojson;
    }

    if (!geometry || !/Polygon$/.test(geometry.type)) {
      return false;
    }

    clearOverlays();
    drawnItems.clearLayers();
    const layer = L.geoJSON(geometry, {
      style: { color: '#FF002B', fillColor: '#FF002B', fillOpacity: 0.2, weight: 3 }
    });
    layer.getLayers().forEach((l) => drawnItems.addLayer(l));

    currentGeoJSON = geometry;
    const bounds = drawnItems.getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });
    }
    if (shapeDrawnCallback) {
      shapeDrawnCallback(currentGeoJSON);
    }
    return true;
  }

  function startDrawingPolygon() {
    if (map && map.pm) {
      map.pm.enableDraw('Polygon', {
        snapping: true,
        continueDrawing: false
      });
    }
  }

  function zoomToIndia() {
    if (map) {
      map.flyTo(INDIA_CENTER, INDIA_ZOOM, { duration: 1.5 });
    }
  }

  function zoomToCampus() {
    if (map) {
      map.flyTo(CAMPUS_CENTER, CAMPUS_ZOOM, { duration: 1.5 });
    }
  }

  // Overlays are single pre-rendered PNGs from the backend, not live GEE XYZ
  // tiles: one HTTP request per layer instead of dozens of per-tile inference
  // calls on every pan/zoom.
  function updateOverlays(rgbOverlay, terrainOverlay) {
    if (!map) return;

    clearOverlays();

    function overlayBounds(b) {
      return [[b.south, b.west], [b.north, b.east]];
    }

    if (rgbOverlay && rgbOverlay.url) {
      rgbTileLayer = L.imageOverlay(rgbOverlay.url, overlayBounds(rgbOverlay.bounds), {
        opacity: 0.8,
        zIndex: 50
      }).addTo(map);
    }

    if (terrainOverlay && terrainOverlay.url) {
      terrainTileLayer = L.imageOverlay(terrainOverlay.url, overlayBounds(terrainOverlay.bounds), {
        opacity: 1.0,
        zIndex: 100
      }).addTo(map);
    }

    // Keep the AOI outline for context, but remove its red fill. Otherwise the
    // map stays red even when the Classified Terrain slider is at 0%.
    setSelectionFill(0);

    // Zoom map bounds to fit drawn shape
    if (drawnItems && drawnItems.getLayers().length > 0) {
      const bounds = drawnItems.getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [50, 50], maxZoom: 16 });
      }
    }
  }

  function setRGBOpacity(opacityVal) {
    if (rgbTileLayer) {
      rgbTileLayer.setOpacity(opacityVal);
    }
  }

  function setTerrainOpacity(opacityVal) {
    if (terrainTileLayer) {
      terrainTileLayer.setOpacity(opacityVal);
    }
  }

  function getCurrentGeoJSON() {
    return currentGeoJSON;
  }

  function getMap() {
    return map;
  }

  return {
    initMap,
    getMap,
    startDrawingPolygon,
    loadGeoJSON,
    zoomToIndia,
    zoomToCampus,
    updateOverlays,
    clearOverlays,
    setRGBOpacity,
    setTerrainOpacity,
    getCurrentGeoJSON
  };
})();
