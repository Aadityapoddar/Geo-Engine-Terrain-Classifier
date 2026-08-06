# frontend-next

The v2 frontend: Vite + TypeScript + MapLibre GL JS.
It talks to the same FastAPI backend as the original `frontend/`, and both can be served side by side.

## Running it

The backend must be up first, since Vite proxies `/api` to it:

```bash
python run_app.py                 # FastAPI on :8000
cd frontend-next && npm install
npm run dev                       # Vite on :5173, proxying /api
```

The proxy target defaults to `http://127.0.0.1:8000`; set `API_URL` if the backend is elsewhere.

```bash
npm run build       # -> dist/
npm run preview     # serve dist/ on :4173
npm run typecheck
```

## Why it is built this way

The original frontend streamed the classified raster as live Earth Engine XYZ tiles.
Every 256px tile is a fresh inference run on Earth Engine, so a viewport needed dozens of them, the map painted in one square at a time, and every pan or zoom paid the whole bill again.
Under load Earth Engine returns HTTP 429 and Leaflet leaves those tiles permanently blank.

Three changes address that:

1. **One pre-rendered overlay instead of N live tiles.** The backend now returns `terrain_overlay`, a single PNG of the classified AOI rendered once in EPSG:3857, plus its bounds. The client drops it on the map as an `image` source. Pan and zoom cost zero further requests.
2. **MapLibre GL instead of Leaflet.** Raster data lives in GPU textures, so zooming rescales what is already on screen rather than tearing down and re-requesting a DOM tile grid.
3. **Nothing is fetched before it can be seen.** The Sentinel-2 RGB layer sat underneath an opaque terrain layer in v1, invisible and still fully fetched. Here it loads only when its opacity slider leaves zero, and its source is capped at zoom 14 - Sentinel-2 is 10 m/px, which is roughly zoom 14 at this latitude, so deeper tiles contain no extra information.

`raster-resampling: nearest` on the terrain layer is a correctness requirement, not a style choice: the overlay encodes class IDs as colours, and bilinear filtering would blend adjacent classes into colours no class owns.

## Dependency notes

MapLibre GL is larger than Leaflet, which the bundle numbers reflect.
Offsetting that, three v1 dependencies are gone: Chart.js (one static donut, now ~40 lines of SVG), leaflet-control-geocoder (now a small Nominatim fetch), and the Google Fonts stylesheet (system font stack).
Terra Draw is dynamically imported on the first click of a draw button, so it stays off the initial load.

See `doc/frontend_performance_report.md` for the measured before/after numbers.
