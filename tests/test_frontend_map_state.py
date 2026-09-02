import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def test_completed_overlay_hides_aoi_fill_and_new_aoi_clears_old_results():
    """The AOI outline may remain, but its red fill must not impersonate output."""
    harness = r"""
const fs = require('fs');
const vm = require('vm');
const source = fs.readFileSync(process.argv[2], 'utf8');
const removed = [];
const layers = [];
const selection = {
  style: {},
  setStyle(next) { Object.assign(this.style, next); }
};
const drawn = {
  clearLayers() { layers.length = 0; },
  addLayer(layer) { layers.push(layer); },
  getLayers() { return layers; },
  eachLayer(callback) { layers.forEach(callback); },
  getBounds() { return { isValid: () => false }; }
};
const map = {
  pm: { addControls() {}, setPathOptions() {}, enableDraw() {} },
  addLayer() {}, removeLayer(layer) { removed.push(layer); },
  on() {}, invalidateSize() {}, flyTo() {}, fitBounds() {}
};
const L = {
    map: () => map,
    control: {
      zoom: () => ({ addTo() {} }),
      layers: () => ({ addTo() {} }),
      geocoder: () => ({ addTo() {} })
    },
    tileLayer: () => ({ addTo() {} }),
    FeatureGroup: function () { return drawn; },
    geoJSON: (_geometry, options) => {
      selection.style = { ...options.style };
      return { getLayers: () => [selection] };
    },
    imageOverlay: (_url, _bounds, options) => ({
      kind: options.zIndex === 100 ? 'terrain' : 'rgb',
      addTo() { return this; },
      setOpacity() {}
    })
};
L.Control = { Geocoder: true, geocoder: L.control.geocoder };
const context = {
  console,
  setTimeout() {},
  window: { addEventListener() {}, L },
  L
};
vm.runInNewContext(source + ';globalThis.controller = MapController;', context);
const controller = context.controller;
controller.initMap('map', () => {});
const geometry = {type: 'Polygon', coordinates: [[[0,0],[1,0],[1,1],[0,0]]]};
controller.loadGeoJSON(geometry);
controller.updateOverlays(
  {url: 'rgb', bounds: {west:0,south:0,east:1,north:1}},
  {url: 'terrain', bounds: {west:0,south:0,east:1,north:1}}
);
if (selection.style.fillOpacity !== 0) {
  throw new Error(`selection fill remains ${selection.style.fillOpacity}`);
}
controller.loadGeoJSON(geometry);
if (!removed.some(layer => layer.kind === 'rgb') ||
    !removed.some(layer => layer.kind === 'terrain')) {
  throw new Error('new AOI retained old result overlays');
}
if (selection.style.fillOpacity !== 0.2) {
  throw new Error(`new selection fill is ${selection.style.fillOpacity}`);
}
"""
    result = subprocess.run(
        ["node", "-", str(REPO / "frontend/js/map.js")],
        input=harness,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
