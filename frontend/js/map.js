// Map Controller Module using Leaflet and Leaflet Geoman

const MapController = (function () {
  let map = null;
  let drawnItems = null;
  let currentGeoJSON = null;

  let rgbTileLayer = null;
  let terrainTileLayer = null;

  const INDIA_CENTER = [20.5937, 78.9629];
  const INDIA_ZOOM = 5;

  const CAMPUS_CENTER = [23.174, 80.026];
  const CAMPUS_ZOOM = 15;

  function initMap(containerId, onShapeDrawnCallback) {
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
      // Remove previous drawn shapes to maintain a single ROI
      drawnItems.clearLayers();
      
      const layer = e.layer;
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

  function updateTileOverlays(rgbUrl, terrainUrl) {
    if (!map) return;

    // Remove existing tile layers if present
    if (rgbTileLayer) {
      map.removeLayer(rgbTileLayer);
      rgbTileLayer = null;
    }
    if (terrainTileLayer) {
      map.removeLayer(terrainTileLayer);
      terrainTileLayer = null;
    }

    if (rgbUrl) {
      rgbTileLayer = L.tileLayer(rgbUrl, {
        maxZoom: 19,
        opacity: 0.8,
        zIndex: 50
      }).addTo(map);
    }

    if (terrainUrl) {
      terrainTileLayer = L.tileLayer(terrainUrl, {
        maxZoom: 19,
        opacity: 1.0,
        zIndex: 100
      }).addTo(map);
    }

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

  return {
    initMap,
    startDrawingPolygon,
    zoomToIndia,
    zoomToCampus,
    updateTileOverlays,
    setRGBOpacity,
    setTerrainOpacity,
    getCurrentGeoJSON
  };
})();
