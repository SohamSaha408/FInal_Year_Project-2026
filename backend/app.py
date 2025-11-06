import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict, List

from predictor_wrapper import predict_from_vedas_ts
from vedas_client import (
    fetch_ndvi_timeseries_point,
    fetch_ndvi_timeseries_polygon,
)

# -------------------------
# Create FastAPI App
# -------------------------
app = FastAPI(
    title="VEDAS Crop Intelligence API",
    description="Fetch NDVI from VEDAS and predict crop yield.",
    version="1.0.0",
)

# -------------------------
# CORS – Allow All (safe for development)
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # You can restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Request Model
# -------------------------
class PredictRequest(BaseModel):
    lat: Optional[float] = None
    lon: Optional[float] = None
    radius_m: int = 30
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    crop_type: Optional[str] = None
    season: Optional[str] = None
    polygon_geojson: Optional[Dict[str, Any]] = None
    kml_text: Optional[str] = None


# -------------------------
# API Endpoint
# -------------------------
@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    P1 Mode:
    - If polygon (GeoJSON or KML) provided → Use polygon NDVI fetch
    - Else → Use lat/lon + radius NDVI fetch
    """
    try:
        # Polygon Case
        if req.polygon_geojson or req.kml_text:
            ts = await fetch_ndvi_timeseries_polygon(
                polygon_geojson=req.polygon_geojson,
                kml_text=req.kml_text,
                start_date=req.start_date,
                end_date=req.end_date,
            )
        # Point Case
        else:
            if req.lat is None or req.lon is None:
                raise HTTPException(status_code=400, detail="lat & lon required if no polygon provided")
            
            ts = await fetch_ndvi_timeseries_point(
                lat=req.lat,
                lon=req.lon,
                radius_m=req.radius_m,
                start_date=req.start_date,
                end_date=req.end_date,
            )

    except Exception as e:
        raise HTTPException(status_code=502, detail=f"VEDAS data fetch failed: {e}")

    if not ts or len(ts) == 0:
        raise HTTPException(status_code=404, detail="No NDVI time series received from VEDAS")

    # Run prediction
    result = predict_from_vedas_ts(ts, crop_type=req.crop_type, season=req.season)

    if result is None or result.get("error"):
        raise HTTPException(status_code=500, detail=result.get("error") if result else "Prediction failed")

    return result


@app.get("/")
def root():
    return {"status": "ok", "message": "VEDAS API is running!"}
