import os
import httpx
from typing import Dict, Any, List

VEDAS_BASE = os.getenv("VEDAS_BASE_URL", "https://vedas.sac.gov.in/vconsole")
API_KEY = os.getenv("VEDAS_API_KEY", "1TVUpE-sbaC72Tj9-yNIpA")

async def fetch_ndvi_timeseries(lat: float, lon: float, radius_m: int = 30, start_date: str | None = None, end_date: str | None = None) -> List[Dict[str, Any]]:
    """
    Fetch NDVI timeseries for a point location via VEDAS API.
    NOTE: Replace the URL/path and query params with the canonical VEDAS endpoint your account provides.
    """
    url = f"{VEDAS_BASE}/api/temporal/location_profile"  # <-- Adjust this to the exact endpoint
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    params = {"lat": lat, "lon": lon, "radius": radius_m}
    if start_date: params["start"] = start_date
    if end_date: params["end"] = end_date

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
        ts = data.get("timeseries", data) if isinstance(data, dict) else data
        # Ensure each item has date + ndvi keys
        normalized = []
        for item in ts:
            date = item.get("date") or item.get("timestamp") or item.get("dt")
            ndvi = item.get("ndvi") or item.get("NDVI") or item.get("value")
            normalized.append({"date": date, "ndvi": ndvi, **{k:v for k,v in item.items() if k not in ("date","timestamp","dt","ndvi","NDVI","value")}})
        return normalized
