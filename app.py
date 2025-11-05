import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vedas_client import fetch_ndvi_timeseries
from predictor_wrapper import predict_from_vedas_ts, try_load_saved_model

app = FastAPI(title="VEDAS NDVI → Yield Prediction API")

class PredictRequest(BaseModel):
    lat: float
    lon: float
    radius_m: int = 30
    start_date: str | None = None
    end_date: str | None = None

@app.on_event("startup")
async def startup_event():
    try_load_saved_model("combined_model.pkl")

@app.post("/api/predict")
async def predict(req: PredictRequest):
    try:
        ts = await fetch_ndvi_timeseries(req.lat, req.lon, req.radius_m, req.start_date, req.end_date)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"VEDAS request failed: {e}")
    if not ts:
        raise HTTPException(status_code=404, detail="No NDVI timeseries returned from VEDAS")
    result = predict_from_vedas_ts(ts)
    if result is None or result.get("error"):
        raise HTTPException(status_code=500, detail=result.get("error") if result else "Prediction failed")
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)
