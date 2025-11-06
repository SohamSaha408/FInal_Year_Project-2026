import os, json
import httpx
from typing import Dict, Any, List, Optional
from fastkml import kml as fastkml
from shapely.geometry import shape, mapping
from shapely.geometry.base import BaseGeometry

VEDAS_BASE = os.getenv("VEDAS_BASE_URL", "https://vedas.sac.gov.in/vconsole")
API_KEY = os.getenv("VEDAS_API_KEY", "")

def _headers():
    return {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

async def fetch_ndvi_timeseries_point(lat: float, lon: float, radius_m: int = 30,
                                      start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    url = f"{VEDAS_BASE}/api/temporal/location_profile"   # TODO: replace with your exact endpoint
    params = {"lat": lat, "lon": lon, "radius": radius_m}
    if start_date: params["start"] = start_date
    if end_date: params["end"] = end_date

    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.get(url, headers=_headers(), params=params)
        r.raise_for_status()
        data = r.json()
        ts = data.get("timeseries", data) if isinstance(data, dict) else data
        return _normalize_ts(ts)

def _kml_to_geojson_polygon(kml_text: str) -> Dict[str, Any]:
    k = fastkml.KML()
    k.from_string(kml_text.encode("utf-8"))
    # take first polygon found
    def _first_geom(feat):
        if hasattr(feat, "geometry") and feat.geometry:
            return feat.geometry
        if hasattr(feat, "features"):
            for f in feat.features():
                g = _first_geom(f)
                if g is not None:
                    return g
        return None
    geom = _first_geom(list(k.features())[0])
    if geom is None:
        raise ValueError("No geometry found in KML")
    shapely_geom: BaseGeometry = shape(json.loads(geom.to_geojson()))
    return {"type": "Feature", "geometry": mapping(shapely_geom), "properties": {}}

async def fetch_ndvi_timeseries_polygon(polygon_geojson: Optional[Dict[str, Any]] = None,
                                        kml_text: Optional[str] = None,
                                        start_date: Optional[str] = None,
                                        end_date: Optional[str] = None) -> List[Dict[str, Any]]:
    if not polygon_geojson and kml_text:
        polygon_geojson = _kml_to_geojson_polygon(kml_text)

    if not polygon_geojson:
        raise ValueError("polygon_geojson or kml_text required")

    url = f"{VEDAS_BASE}/api/temporal/polygon_profile"    # TODO: replace with your exact endpoint
    payload = {"polygon": polygon_geojson}
    if start_date: payload["start"] = start_date
    if end_date: payload["end"] = end_date

    async with httpx.AsyncClient(timeout=90.0) as client:
        r = await client.post(url, headers=_headers(), json=payload)
        r.raise_for_status()
        data = r.json()
        ts = data.get("timeseries", data) if isinstance(data, dict) else data
        return _normalize_ts(ts)

def _normalize_ts(ts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    norm = []
    for item in ts:
        date = item.get("date") or item.get("timestamp") or item.get("dt")
        ndvi = item.get("ndvi") or item.get("NDVI") or item.get("value")
        # optional extras if VEDAS returns them
        lst = item.get("lst") or item.get("LST")
        rain = item.get("rain") or item.get("rainfall") or item.get("Rainfall")
        sm = item.get("soil_moisture") or item.get("sm") or item.get("Soil_Moisture")
        norm.append({ "date": date, "ndvi": ndvi, "lst": lst, "rainfall": rain, "soil_moisture": sm })
    return norm
