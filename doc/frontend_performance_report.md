# Frontend performance: v1 (Leaflet) vs v2 (MapLibre)

Measured 2026-08-06. AOI: 5.3 x 5.3 km over the Jabalpur campus study area, Random Forest, 2025-03-01 to 2025-04-30, cloud < 15%, smoothing on.
Network timings are direct HTTP fetches capped at six concurrent connections, which is what a browser allows per host.
Browser timings are Playwright driving Chromium at 1440x900, both frontends given the identical `/api/classify` response and the same scripted camera moves, each driven through its own map code.
Chromium renders WebGL in software here, so v2's GPU path is under-represented and the rendering wins below are a floor.

## Diagnosis

The classified raster was streamed as live Earth Engine XYZ tiles.
Every 256px tile re-runs the whole chain: composite, SAR, GLCM texture, classify.
It is not a cached image being sliced up, it is inference on demand, once per tile.

| Live tile cost (6 parallel) | Tiles | Wall | Median/tile | Undelivered |
|---|---|---|---|---|
| One cold z15 tile | 1 | 5.17 s | 5.17 s | 0 |
| z14 viewport | 12 | 6.26 s | 3.82 s | 6 |
| z15 viewport | 36 | 28.73 s | 3.95 s | 22 |
| z16 viewport | 121 | 114.57 s | 3.95 s | 74 |

65 of those 121 z16 tiles returned HTTP 429, `Exceeded Earth Engine concurrency limit. Your project is in Restricted Mode.`
Leaflet does not retry a failed tile: it fires `tileerror` and leaves the square blank permanently.
That is the reported symptom exactly. See `doc/assets/frontend_v1_tile_holes.png` against `doc/assets/frontend_v2_overlay.png`, same AOI, same zoom.

Three things compounded it:

- **maxZoom 19 with no native-resolution cap.** Sentinel-2 is 10 m/px, roughly zoom 14 at this latitude, so zooming 14 to 17 re-requested the whole viewport four more times at full inference cost for no extra information.
- **The RGB layer was fetched but never visible,** sitting at 80% opacity under a 100%-opaque classified layer.
- **No progress feedback** during a request that can legitimately take two minutes.

## Numbers

### Page load

| | v1 | v2 | |
|---|---|---|---|
| Time to `load` | 3408 ms | 638 ms | 5.3x faster |
| First Contentful Paint | 832 ms | 472 ms | 1.8x faster |
| App + vendor requests | 16 | 5 | 3.2x fewer |
| Third-party asset origins | 4 | 0 | unpkg, jsdelivr, fonts.googleapis, fonts.gstatic |

### Painting the classified layer

Median of three spaced runs, plus the in-browser run.

| | v1 (36 live tiles) | v2 (1 overlay) |
|---|---|---|
| Earth Engine requests | 36 | 1 |
| Delivered | 19 of 36 (53%) | 1 of 1 (100%) |
| Wall to done | 19.9 s (range 19.5-22.6) | 19.7 s (range 17.4-25.2) |
| In-browser requests / failed | 50 / 46 (92%) | 1 / 0 |
| In-browser first pixels | 2.2 s | 13.9 s |
| Transfer | 78 KB, incomplete | 38 KB, complete |

**Wall-clock time is a wash, and that is the expected result.**
Both paths classify the same pixels over the same area, so the inference cost is identical.
Splitting it into 36 requests does not make it cheaper, it adds 36x the request overhead and puts the job on the wrong side of a concurrency limit.
What changes is what you get for the wait: a complete layer instead of roughly half of one.

v1 shows its first square sooner (2.2 s vs 13.9 s) because it paints tiles as they trickle in.
That is a real regression in time-to-first-something; v2 now holds a progress readout with a live elapsed counter until the raster is actually on screen.

### Zoom sweep, z13 to z17

This is where the architectures separate.

| | v1 | v2 |
|---|---|---|
| Total requests | 316 | 89 |
| Earth Engine tile requests | 226 | 0 |
| Delivered / failed | 15 / 211 (93% fail) | n/a |
| Bytes | 1470 KB | 1149 KB |

v2 issues no Earth Engine request during the sweep at all.
The overlay is fetched once and every subsequent zoom is a GPU rescale of a texture already in memory.
v1 pays for the viewport again at every zoom level, and at 93% failure most of that buys nothing.

### Frame timing

| | v1 | v2 |
|---|---|---|
| Median frame | 16.7 ms | 16.7 ms |
| p95 frame | 19.3 ms | 18.7 ms |
| Frames over 50 ms | 5 of 1088 (0.5%) | 5 of 1053 (0.5%) |

Frame rate was never the bottleneck.
Both hold 60 fps.
The lag was latency and missing tiles, and nothing that only made rendering faster would have fixed it.

### Bundle (gzipped, measured from the files)

| | v1 | v2 |
|---|---|---|
| Vendor | 199.8 KB across 7 CDN files | bundled |
| Own JS + CSS + HTML | 13.5 KB | bundled |
| Critical path | 213.3 KB plus Inter webfont files | 261.3 KB |
| Deferred | none | 44.3 KB (Terra Draw, on first draw click) |

v2's critical bundle is ~48 KB larger, and that is a real regression.
MapLibre GL is heavier than Leaflet and trimming elsewhere did not fully cover it; dropping Chart.js (69.6 KB), leaflet-control-geocoder (11.4 KB) and the Google Fonts request recovered most of the gap.
It is a one-time cost on a cold cache against a per-interaction cost that used to be paid forever.

## What changed in v2

1. **One pre-rendered overlay instead of N live tiles.** The backend renders the AOI once via `getThumbURL` in EPSG:3857 and returns it with its bounds; the client adds it as an `image` source. A 5.3 km AOI is 532 px of real data at native 10 m, so the whole layer is 38 KB. Requested at native resolution, capped at 2048 px.
2. **MapLibre GL instead of Leaflet.** Raster lives in GPU textures, so zoom rescales what is on screen instead of re-requesting a DOM tile grid.
3. **Nothing fetched before it can be seen.** RGB loads only when its opacity slider leaves zero, and its source is capped at zoom 14.
4. **`raster-resampling: nearest`.** The overlay encodes class IDs as colours; bilinear filtering would blend Forest into Water and show a class the model never predicted. A correctness fix.
5. **Progress readout** with elapsed seconds and a stage label, held until the raster is on screen.

## What did not improve, and what is open

`/api/classify` is unchanged and remains the dominant wait: ~8.7 s warm, up to 140 s the first time a model and date window are trained.
That is Earth Engine compute, not frontend work.
The only backend change was adding the overlay to the response, with its bounding box computed locally rather than via an extra Earth Engine round trip.

The Sentinel-2 RGB layer still streams live tiles and still draws 429s when a user turns it on.
It is now opt-in and capped at native zoom, so it is off the default path rather than fixed; giving it the same pre-rendered treatment as the classified layer is the obvious next step.

Time-to-first-pixel is worse.
A low-resolution overlay fetched first and swapped for the full one would address it, but it is not obviously a win: the cost is inference over the AOI, not pixel count, so a smaller thumbnail may not return meaningfully sooner. Worth measuring before building.

## Caveats

The 429 rate depends on the project's Earth Engine quota tier, and this project is in Restricted Mode.
Higher concurrency would fail fewer tiles but would still pay one inference per tile per zoom level.
The architecture creates the exposure; the quota decides how visible it is.

Earth Engine latency varies substantially between runs, which is why the paint table reports a median of three spaced runs and a range.
Request counts and failure rates were stable across runs; wall-clock seconds were not.
