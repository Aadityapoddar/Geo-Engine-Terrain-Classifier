"""Checks for the static-overlay geometry maths.

These decide where the classified PNG lands on the map, so getting them wrong
misplaces every pixel rather than failing loudly.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.gee_classifier import _iter_coords


class TestIterCoords(unittest.TestCase):

    def test_polygon_ring(self):
        poly = [[[80.0, 23.15], [80.05, 23.15], [80.05, 23.20], [80.0, 23.20], [80.0, 23.15]]]
        self.assertEqual(len(list(_iter_coords(poly))), 5)

    def test_bbox_of_polygon(self):
        poly = [[[80.0, 23.15], [80.05, 23.15], [80.05, 23.20], [80.0, 23.20], [80.0, 23.15]]]
        pts = list(_iter_coords(poly))
        lons = [p[0] for p in pts]
        lats = [p[1] for p in pts]
        self.assertEqual((min(lons), max(lons)), (80.0, 80.05))
        self.assertEqual((min(lats), max(lats)), (23.15, 23.20))

    def test_multipolygon_nesting(self):
        """One extra level of nesting must not change the bounding box."""
        multi = [
            [[[80.0, 23.15], [80.05, 23.15], [80.0, 23.20], [80.0, 23.15]]],
            [[[80.10, 23.25], [80.15, 23.25], [80.10, 23.30], [80.10, 23.25]]],
        ]
        pts = list(_iter_coords(multi))
        lons = [p[0] for p in pts]
        self.assertEqual((min(lons), max(lons)), (80.0, 80.15))

    def test_point(self):
        self.assertEqual(list(_iter_coords([80.0, 23.15])), [(80.0, 23.15)])


class TestOverlayResolution(unittest.TestCase):
    """The px calculation should ask for native 10 m resolution, then stop."""

    @staticmethod
    def px_for(west, east, lat, max_px=2048):
        width_m = (east - west) * 111320.0 * math.cos(math.radians(lat))
        return max(256, min(max_px, int(width_m / 10)))

    def test_small_aoi_gets_native_resolution(self):
        # ~5.3 km at 23N -> ~530 px, i.e. one pixel per Sentinel-2 pixel.
        self.assertAlmostEqual(self.px_for(80.0, 80.052, 23.17), 532, delta=5)

    def test_large_aoi_is_capped(self):
        self.assertEqual(self.px_for(80.0, 81.0, 23.17), 2048)

    def test_tiny_aoi_has_a_floor(self):
        self.assertEqual(self.px_for(80.0, 80.001, 23.17), 256)


if __name__ == "__main__":
    unittest.main()
