# app.py (only showing changed parts)
from pydantic import BaseModel
from typing import Optional, Any, Dict
from vedas_client import (
    fetch_ndvi_timeseries_point,
    fetch_ndvi_timeseries_polygon,
)
from predictor_wrapper import predict_from_vedas_ts


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

@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        # Choose polygon if provided; else point+radius
        if req.polygon_geojson or req.kml_text:
            ts = await fetch_ndvi_timeseries_polygon(
                polygon_geojson=req.polygon_geojson,
                kml_text=req.kml_text,
                start_date=req.start_date,
                end_date=req.end_date,
            )
        else:
            if req.lat is None or req.lon is None:
                raise HTTPException(status_code=400, detail="lat/lon required when polygon not provided")
            ts = await fetch_ndvi_timeseries_point(
                lat=req.lat, lon=req.lon, radius_m=req.radius_m,
                start_date=req.start_date, end_date=req.end_date
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"VEDAS request failed: {e}")

    if not ts:
        raise HTTPException(status_code=404, detail="No NDVI timeseries returned from VEDAS")

    result = predict_from_vedas_ts(ts, crop_type=req.crop_type, season=req.season)
    if result is None or result.get("error"):
        raise HTTPException(status_code=500, detail=result.get("error") if result else "Prediction failed")
    return result
