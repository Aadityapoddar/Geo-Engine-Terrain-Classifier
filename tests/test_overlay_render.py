"""The overlay must be rendered on the grid the classifier was trained on.

The bug this guards against: the terrain overlay used to be a single thumbnail
capped at 2048 px, so a district was rendered at ~60 m/px. Earth Engine
re-evaluates the whole chain at whatever scale it is asked for, and two of the
nineteen features are GLCM textures over a 3x3 *pixel* window, so the model was
answering at a scale it had never been trained on. Measured on Jabalpur, the
same request reported Water at 5.65% of the AOI and painted it over 36% of the
map.

These are pure grid arithmetic -- no Earth Engine, no network.
"""
import math
import unittest
import urllib.error

from backend.gee_classifier import (
    TILE_PX,
    _is_rate_limited,
    _render_grid,
    _tile_boxes,
    overlay_cache_key,
)

# Jabalpur district's bounding box, the AOI the bug was reported on.
JABALPUR_BBOX = {
    "type": "Polygon",
    "coordinates": [[
        [79.4585, 22.8426], [80.7395, 22.8426],
        [80.7395, 23.4823], [79.4585, 23.4823], [79.4585, 22.8426],
    ]],
}


class TestRenderGrid(unittest.TestCase):
    def test_grid_is_ten_ground_metres_per_pixel(self):
        """Not 10 Web-Mercator metres -- those are ~9.2 m of ground at this latitude."""
        grid = _render_grid(JABALPUR_BBOX)

        lat_mid = math.radians((grid["north"] + grid["south"]) / 2)
        ground_width_m = (grid["east"] - grid["west"]) * 111320.0 * math.cos(lat_mid)
        metres_per_px = ground_width_m / grid["width"]

        self.assertAlmostEqual(metres_per_px, 10.0, delta=0.15)

    def test_a_district_needs_more_than_one_tile(self):
        """If this ever fits in one tile the AOI is being silently downscaled."""
        grid = _render_grid(JABALPUR_BBOX)
        self.assertGreater(grid["width"], TILE_PX)
        self.assertGreater(len(_tile_boxes(grid)), 1)


class TestTiling(unittest.TestCase):
    def setUp(self):
        self.grid = _render_grid(JABALPUR_BBOX)
        self.boxes = _tile_boxes(self.grid)

    def test_tiles_cover_the_grid_exactly(self):
        """Every pixel rendered once: a gap is a hole in the map, an overlap is a seam."""
        covered = sum(
            (right - left) * (bottom - top)
            for left, top, right, bottom in (b["px"] for b in self.boxes)
        )
        self.assertEqual(covered, self.grid["width"] * self.grid["height"])

    def test_no_tile_overlaps_another(self):
        seen = set()
        for left, top, right, bottom in (b["px"] for b in self.boxes):
            key = (left, top)
            self.assertNotIn(key, seen)
            seen.add(key)
            self.assertLessEqual(right, self.grid["width"])
            self.assertLessEqual(bottom, self.grid["height"])

    def test_tile_regions_touch_without_gaps(self):
        """Adjacent tiles must share an edge, or the stitched PNG is misregistered."""
        by_origin = {b["px"][:2]: b for b in self.boxes}
        for (left, top), box in by_origin.items():
            right_neighbour = by_origin.get((box["px"][2], top))
            if right_neighbour:
                self.assertAlmostEqual(
                    box["region"][2], right_neighbour["region"][0], places=9)
            below = by_origin.get((left, box["px"][3]))
            if below:
                self.assertAlmostEqual(
                    box["region"][1], below["region"][3], places=9)

    def test_tiles_span_the_whole_aoi(self):
        west = min(b["region"][0] for b in self.boxes)
        east = max(b["region"][2] for b in self.boxes)
        south = min(b["region"][1] for b in self.boxes)
        north = max(b["region"][3] for b in self.boxes)
        self.assertAlmostEqual(west, self.grid["west"], places=6)
        self.assertAlmostEqual(east, self.grid["east"], places=6)
        self.assertAlmostEqual(south, self.grid["south"], places=6)
        self.assertAlmostEqual(north, self.grid["north"], places=6)


class TestCacheKey(unittest.TestCase):
    BASE = (JABALPUR_BBOX, "gtb", "2025-04-01", "2025-05-30", 15.0, True)

    def test_same_request_same_key(self):
        self.assertEqual(overlay_cache_key(*self.BASE), overlay_cache_key(*self.BASE))

    def test_every_input_changes_the_key(self):
        """A cached PNG must never outlive the request that produced it."""
        variants = [
            ({"type": "Polygon", "coordinates": [[[79.0, 22.0], [79.1, 22.0],
                                                  [79.1, 22.1], [79.0, 22.0]]]},
             "gtb", "2025-04-01", "2025-05-30", 15.0, True),
            (JABALPUR_BBOX, "rf", "2025-04-01", "2025-05-30", 15.0, True),
            (JABALPUR_BBOX, "gtb", "2025-01-01", "2025-05-30", 15.0, True),
            (JABALPUR_BBOX, "gtb", "2025-04-01", "2025-02-28", 15.0, True),
            (JABALPUR_BBOX, "gtb", "2025-04-01", "2025-05-30", 30.0, True),
            (JABALPUR_BBOX, "gtb", "2025-04-01", "2025-05-30", 15.0, False),
        ]
        base = overlay_cache_key(*self.BASE)
        for variant in variants:
            self.assertNotEqual(base, overlay_cache_key(*variant))



class TestRateLimitDetection(unittest.TestCase):
    """A 25-minute render must not be thrown away because Earth Engine was busy.

    The two calls in a tile fail differently -- `getThumbURL` raises
    `EEException`, the download that follows raises `urllib.HTTPError` -- and
    only one of them was being retried.
    """

    def test_ee_concurrency_error_is_retryable(self):
        ex = Exception(
            "Too Many Requests: Exceeded Earth Engine concurrency limit. "
            "Your project is in Restricted Mode.")
        self.assertTrue(_is_rate_limited(ex))

    def test_http_429_is_retryable(self):
        ex = urllib.error.HTTPError("http://ee", 429, "Too Many Requests", {}, None)
        self.assertTrue(_is_rate_limited(ex))

    def test_real_failures_are_not_retried(self):
        for ex in (
            urllib.error.HTTPError("http://ee", 400, "Bad Request", {}, None),
            Exception("Reprojection output too large (13726x7736 pixels)."),
            Exception("User memory limit exceeded."),
        ):
            self.assertFalse(_is_rate_limited(ex))


if __name__ == "__main__":
    unittest.main()
