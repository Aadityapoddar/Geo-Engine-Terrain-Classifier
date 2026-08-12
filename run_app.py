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

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print(f"📡 Serving Web Frontend & API Backend at: {url}")
    print("✨ Features:")
    print("   • Default Map Center: India (20.5937, 78.9629)")
    print("   • Models: Random Forest, SVM, XGBoost, CART, KNN")
    classes = ", ".join(c["name"] for c in LAND_COVER_CLASSES.values())
    print(f"   • Individual Class Areas: {classes}")
    print("   • Exports: GeoTIFF downloads & GeoJSON reports")
    print("=" * 65)

    # Attempt to launch default web browser
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
