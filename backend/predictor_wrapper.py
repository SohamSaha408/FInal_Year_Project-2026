import pandas as pd
from typing import Optional
from final import NDVIYieldPredictor
PREDICTOR = NDVIYieldPredictor()

def _timeseries_to_df(timeseries):
    rows = []
    for entry in timeseries:
        date_str = entry.get("date")
        dt = pd.to_datetime(date_str)
        date_label = dt.strftime("%b-%d")
        rows.append({"DateTime": date_label, "YR": entry.get("ndvi", None)})
    return pd.DataFrame(rows)

def predict_from_vedas_ts(timeseries, crop_type: Optional[str] = None, season: Optional[str] = None):
    df = _timeseries_to_df(timeseries)
    if df is None or df.empty:
        return {"error": "Empty VEDAS timeseries"}

    processed = PREDICTOR.process_vedas_data(df)
    if processed is None or len(processed) < 5:
        return {"error": "Insufficient processed samples for prediction"}

    # Allow overriding categorical features if provided
    if crop_type:
        mapping = {"rice":1, "wheat":2, "maize":3, "soybean":4, "cotton":5}
        processed["Crop_Type"] = mapping.get(crop_type.lower(), processed["Crop_Type"])
    if season:
        mapping = {"kharif":1, "rabi":2, "zaid":3}
        processed["Season"] = mapping.get(season.lower(), processed["Season"])

    y_test, y_pred = PREDICTOR.train_model(processed)
    avg_ndvi = float(processed['NDVI'].mean())
    avg_conditions = [
        avg_ndvi,
        avg_ndvi * 0.8,
        float(processed['LST'].mean()),
        float(processed['Rainfall'].mean()),
        float(processed['Soil_Moisture'].mean()),
        int(processed['Season'].mode()[0]),
        int(processed['Crop_Type'].mode()[0])
    ]
    predicted_yield = float(PREDICTOR.predict_yield(avg_conditions))

    try:
        import numpy as np
        conf = float(np.mean((y_test - y_pred) ** 2)) if y_test is not None else None
    except Exception:
        conf = None

    return {
        "predicted_yield": predicted_yield,
        "avg_ndvi": avg_ndvi,
        "confidence_mse": conf,
        "ndvi_timeseries": timeseries,
        "meta": {"samples_used": int(len(processed))}
    }
