FROM python:3.11-slim

WORKDIR /app

# ponytail: requirements.txt is the dev/notebook superset (geemap, seaborn, folium
# pull in matplotlib and pandas). The backend imports only these five. Add here if
# backend/ ever grows an import.
RUN pip install --no-cache-dir \
    earthengine-api \
    python-dotenv \
    fastapi \
    "uvicorn[standard]" \
    pydantic

COPY backend/ backend/
COPY frontend/ frontend/

# Cloud Run injects PORT; 8000 keeps `docker run -p 8000:8000` working locally.
CMD exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
