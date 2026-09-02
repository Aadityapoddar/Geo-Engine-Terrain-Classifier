/**
 * Export the drawn `barren_points` geometry as a labelled Barren Land asset.
 *
 * Paste this at the bottom of the `jabalpur_barren_land_points` script in the
 * Earth Engine Code Editor, press Run, then start the task from the Tasks tab.
 *
 * Why the explicit `label`: a Geometry Import exports with no attributes at all.
 * `jabalpur_agriculture_points_updated` was created that way and every feature
 * came out with empty properties, so training dropped all 1,000 points in
 * silence and the Agriculture class was never learned. Setting the label here is
 * what stops that repeating for Barren Land.
 *
 * Class 3 is Barren Land -- the slot the two retired Soil assets used to fill.
 * Agriculture is 4; Sand was dropped in v3.
 */

var BARREN_LABEL = 3;
// The 2026-08-08 export ran with this id (task export_jabalpur_barren_points_asset).
var ASSET_ID = 'users/ashutoshsaxena703/jabalpur_barren_points';

// `barren_points` is the drawn layer; ee.FeatureCollection() accepts it directly.
var labelled = ee.FeatureCollection(barren_points).map(function (feature) {
  return ee.Feature(feature.geometry(), {label: BARREN_LABEL});
});

print('barren points to export:', labelled.size());
print('label histogram:', labelled.aggregate_histogram('label'));

Export.table.toAsset({
  collection: labelled,
  description: 'barren_land_points_labelled',
  assetId: ASSET_ID
});
