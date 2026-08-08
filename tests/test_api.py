import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import app


class TestGeoEngineAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertTrue(data["gee_initialized"])
        self.assertEqual(data["default_map_center"]["label"], "India")

    def test_models_endpoint(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("rf", data["models"])
        self.assertIn("svm", data["models"])
        self.assertIn("xgb", data["models"])
        self.assertIn("cart", data["models"])
        self.assertIn("knn", data["models"])

    def test_classification_endpoint(self):
        # Campus study area polygon
        test_geojson = {
            "type": "Polygon",
            "coordinates": [
                [
                    [80.0171, 23.1739],
                    [80.0325, 23.1653],
                    [80.0365, 23.1725],
                    [80.0268, 23.1816],
                    [80.0154, 23.1769],
                    [80.0171, 23.1739]
                ]
            ]
        }

        payload = {
            "geometry": test_geojson,
            "model_type": "rf",
            "start_date": "2025-03-01",
            "end_date": "2025-04-30",
            "cloud_threshold": 15.0,
            "smoothing": True
        }

        response = self.client.post("/api/classify", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "success")
        self.assertIn("tile_urls", data)
        self.assertIn("sentinel_rgb", data["tile_urls"])
        self.assertIn("terrain_classified", data["tile_urls"])
        
        # Verify individual class areas breakdown
        self.assertIn("individual_class_areas", data)
        class_areas = data["individual_class_areas"]
        self.assertEqual(len(class_areas), 5)

        for item in class_areas:
            self.assertIn("name", item)
            self.assertIn("area_ha", item)
            self.assertIn("area_sqm", item)
            self.assertIn("percentage", item)

        print("\n✅ Classification Test Result:")
        print(f"Total Area: {data['summary']['total_area_ha']} ha")
        for cls in class_areas:
            print(f" - {cls['name']}: {cls['area_ha']} ha ({cls['percentage']}%)")


if __name__ == "__main__":
    unittest.main()
