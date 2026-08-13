#!/usr/bin/env python3
import os
import sys
import webbrowser
import time
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from backend.config import LAND_COVER_CLASSES  # noqa: E402

def main():
    print("=" * 65)
    print("🚀 Starting Geo-Engine Terrain Classifier Web Application")
    print("=" * 65)

    ee_project = os.getenv("EE_PROJECT_ID")
    print(f"🌍 Earth Engine Project ID: {ee_project}")

    # Render (and most PaaS) set PORT and scan 0.0.0.0. Local runs stay on loopback.
    hosted = bool(os.getenv("RENDER") or os.getenv("PORT"))
    host = "0.0.0.0" if hosted else "127.0.0.1"
    port = int(os.getenv("PORT", "8000"))
    url = f"http://{host}:{port}"

    print(f"📡 Serving Web Frontend & API Backend at: {url}")
    print("✨ Features:")
    print("   • Default Map Center: India (20.5937, 78.9629)")
    print("   • Models: Random Forest, SVM, XGBoost, CART, KNN")
    classes = ", ".join(c["name"] for c in LAND_COVER_CLASSES.values())
    print(f"   • Individual Class Areas: {classes}")
    print("   • Exports: GeoTIFF downloads & GeoJSON reports")
    print("=" * 65)

    if not hosted:
        try:
            def open_browser():
                time.sleep(1.5)
                webbrowser.open(url)

            import threading
            threading.Thread(target=open_browser, daemon=True).start()
        except Exception as e:
            print(f"Note: Automatic browser opening skipped ({e}).")

    uvicorn.run("backend.app:app", host=host, port=port, reload=False)

if __name__ == "__main__":
    main()
