import React, { useMemo, useState } from 'react'
import axios from 'axios'
import { Leaf, MapPin, Upload, Calendar, LineChart as IconLine, Thermometer, CloudRain, Download } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend } from 'recharts'

const BACKEND_URL = "https://final-year-project-2026-y128.onrender.com" // hard-coded backend

const readFileAsText = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader();
  reader.onload = () => resolve(reader.result);
  reader.onerror = reject;
  reader.readAsText(file);
});

export default function App(){
  // form
  const [lat, setLat] = useState(22.5726)
  const [lon, setLon] = useState(88.3639)
  const [radius, setRadius] = useState(30)
  const [crop, setCrop] = useState('rice')
  const [season, setSeason] = useState('kharif')
  const [dateFrom, setDateFrom] = useState('2025-05-01')
  const [dateTo, setDateTo] = useState('2025-11-01')
  const [boundaryFile, setBoundaryFile] = useState(null)

  // data
  const [series, setSeries] = useState([])
  const [predictedYield, setPredictedYield] = useState(null)
  const [confidence, setConfidence] = useState(null)
  const [raw, setRaw] = useState(null)
  const [loading, setLoading] = useState(false)

  const avgNDVI = useMemo(() => {
    if(!series?.length) return 0
    return series.reduce((a,b)=> a + (Number(b.ndvi)||0), 0) / series.length
  }, [series])

  const health = useMemo(()=> avgNDVI > 0.55 ? 'Good' : avgNDVI > 0.4 ? 'Moderate' : 'Poor', [avgNDVI])

  const buildPayload = async () => {
    const payload = {
      lat: Number(lat),
      lon: Number(lon),
      radius_m: Number(radius),
      start_date: dateFrom || null,
      end_date: dateTo || null,
      crop_type: crop,
      season: season
    }
    if(boundaryFile){
      const ext = boundaryFile.name.toLowerCase()
      const text = await readFileAsText(boundaryFile)
      if(ext.endsWith('.geojson') || ext.endsWith('.json')){
        try{
          payload.polygon_geojson = JSON.parse(text)
        }catch(e){ throw new Error('Invalid GeoJSON file') }
      }else if(ext.endsWith('.kml')){
        payload.kml_text = text
      }
    }
    return payload
  }

  const fetchAndPredict = async () => {
    setLoading(True)
    try{
      const url = `${BACKEND_URL.replace(/\/$/,'')}/api/predict`
      const payload = await buildPayload()
      const res = await axios.post(url, payload, { timeout: 60000 })
      const data = res.data || {}
      setSeries(data.ndvi_timeseries || [])
      setPredictedYield(data.predicted_yield ?? null)
      setConfidence(data.confidence_mse ?? null)
      setRaw(data)
    }catch(err){
      console.error(err)
      alert(err?.response?.data?.detail || err.message || 'Failed to fetch from backend')
    }finally{
      setLoading(false)
    }
  }

  const downloadCSV = () => {
    const header = 'date,ndvi,lst,rainfall,soil_moisture\n'
    const body = (series||[]).map(r => `${r.date},${Number(r.ndvi ?? '').toFixed?.(3) || ''},${Number(r.lst ?? '').toFixed?.(2) || ''},${Number(r.rainfall ?? r.rain ?? '').toFixed?.(1) || ''},${Number(r.soil_moisture ?? r.sm ?? '').toFixed?.(3) || ''}`).join('\n')
    const blob = new Blob([header + body], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'vedas_timeseries.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  // simple tabs
  const [tab, setTab] = useState('ndvi')

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-slate-800 text-slate-100 p-6">
      <header className="max-w-7xl mx-auto flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <Leaf className="w-7 h-7" />
          <h1 className="text-2xl md:text-3xl font-semibold tracking-tight">VEDAS Pro — Crop Intelligence</h1>
          <span className="ml-2 text-xs px-2 py-1 rounded bg-emerald-600/20 border border-emerald-500/30">Live (Real Data)</span>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={fetchAndPredict} disabled={loading} className="rounded-2xl shadow px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60">{loading ? 'Working…' : 'Fetch & Predict'}</button>
          <button onClick={downloadCSV} className="rounded-2xl shadow px-4 py-2 bg-slate-700 hover:bg-slate-600 flex items-center gap-2"><Download className="w-4 h-4"/>Export CSV</button>
        </div>
      </header>

      <div className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Sidebar */}
        <div className="col-span-1 lg:col-span-4 rounded-2xl bg-slate-900/60 border border-slate-700/60 p-5 space-y-4">
          <div className="flex items-center gap-2 text-slate-300">
            <MapPin className="w-4 h-4"/><p className="text-sm">Field Inputs</p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-slate-300 text-sm">Latitude</label>
              <input type="number" step="0.0001" value={lat} onChange={e=>setLat(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70"/>
            </div>
            <div>
              <label className="text-slate-300 text-sm">Longitude</label>
              <input type="number" step="0.0001" value={lon} onChange={e=>setLon(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70"/>
            </div>
            <div>
              <label className="text-slate-300 text-sm">Radius (m)</label>
              <input type="number" value={radius} onChange={e=>setRadius(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70"/>
            </div>
            <div>
              <label className="text-slate-300 text-sm">Crop Type</label>
              <select value={crop} onChange={e=>setCrop(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70">
                <option value="rice">Rice</option>
                <option value="wheat">Wheat</option>
                <option value="maize">Maize</option>
                <option value="soybean">Soybean</option>
                <option value="cotton">Cotton</option>
              </select>
            </div>
            <div>
              <label className="text-slate-300 text-sm">Season</label>
              <select value={season} onChange={e=>setSeason(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70">
                <option value="kharif">Kharif</option>
                <option value="rabi">Rabi</option>
                <option value="zaid">Zaid</option>
              </select>
            </div>
            <div className="col-span-2 grid grid-cols-2 gap-3">
              <div>
                <label className="text-slate-300 text-sm flex items-center gap-2"><Calendar className="w-4 h-4"/>From</label>
                <input type="date" value={dateFrom} onChange={e=>setDateFrom(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70"/>
              </div>
              <div>
                <label className="text-slate-300 text-sm flex items-center gap-2"><Calendar className="w-4 h-4"/>To</label>
                <input type="date" value={dateTo} onChange={e=>setDateTo(e.target.value)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70"/>
              </div>
            </div>
            <div className="col-span-2">
              <label className="text-slate-300 text-sm flex items-center gap-2"><Upload className="w-4 h-4"/> Upload Field Boundary (GeoJSON/KML)</label>
              <input type="file" accept=".geojson,.json,.kml" onChange={e=>setBoundaryFile(e.target.files?.[0] || null)} className="w-full mt-1 px-3 py-2 rounded bg-slate-800/70"/>
              {boundaryFile && <p className="text-xs text-slate-400 mt-1">Uploaded: {boundaryFile.name}</p>}
            </div>
          </div>
          <div className="flex gap-3 pt-1">
            <button onClick={fetchAndPredict} disabled={loading} className="rounded-xl w-full px-4 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60">{loading ? 'Fetching…' : 'Fetch & Predict'}</button>
            <button onClick={downloadCSV} className="rounded-xl w-full px-4 py-2 bg-slate-700 hover:bg-slate-600">Export CSV</button>
          </div>
        </div>

        {/* Main content */}
        <div className="col-span-1 lg:col-span-8 space-y-6">
          {/* KPI cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
              <p className="text-sm text-slate-400">Predicted Yield</p>
              <p className="text-3xl font-semibold mt-1">{predictedYield != null ? Number(predictedYield).toLocaleString() : '—'} kg/ha</p>
              <p className="text-xs text-slate-500 mt-2">Returned by backend from your model</p>
            </div>
            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
              <p className="text-sm text-slate-400">Crop Health</p>
              <div className="mt-1 flex items-center gap-2">
                <span className="rounded-full border border-emerald-500/40 text-emerald-300 px-2 py-0.5 text-xs">{series?.length ? health : '—'}</span>
                <span className="text-xs text-slate-400">Avg NDVI {series?.length ? avgNDVI.toFixed(3) : '—'}</span>
              </div>
              <p className="text-xs text-slate-500 mt-2">Higher NDVI indicates better vigor</p>
            </div>
            <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
              <p className="text-sm text-slate-400">Model Confidence</p>
              <p className="text-3xl font-semibold mt-1">{confidence != null ? Math.round(Number(confidence) * 100) + '%' : '—'}</p>
              <p className="text-xs text-slate-500 mt-2">Backend MSE/R² surrogate</p>
            </div>
          </div>

          {/* Tabs */}
          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="inline-flex rounded-xl overflow-hidden bg-slate-800/60">
              <button onClick={()=>setTab('ndvi')} className={`px-4 py-2 text-sm ${tab==='ndvi'?'bg-emerald-600 text-white':'text-slate-300 hover:bg-slate-700/50'}`}><IconLine className="w-4 h-4 inline-block mr-1"/>NDVI</button>
              <button onClick={()=>setTab('lst')} className={`px-4 py-2 text-sm ${tab==='lst'?'bg-emerald-600 text-white':'text-slate-300 hover:bg-slate-700/50'}`}><Thermometer className="w-4 h-4 inline-block mr-1"/>LST</button>
              <button onClick={()=>setTab('rain')} className={`px-4 py-2 text-sm ${tab==='rain'?'bg-emerald-600 text-white':'text-slate-300 hover:bg-slate-700/50'}`}><CloudRain className="w-4 h-4 inline-block mr-1"/>Rainfall</button>
            </div>

            {tab === 'ndvi' && (
              <div className="h-64 mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="ndvi" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {tab === 'lst' && (
              <div className="h-64 mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey="lst" dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}

            {tab === 'rain' && (
              <div className="h-64 mt-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Legend />
                    <Line type="monotone" dataKey={(series[0] && ('rainfall' in series[0])) ? 'rainfall' : 'rain'} dot={false} strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

          {/* Raw JSON */}
          <div className="bg-slate-900/60 border border-slate-700/60 rounded-2xl p-5">
            <div className="flex items-center gap-2 mb-3 text-slate-300">
              <code className="text-xs bg-slate-800/70 px-2 py-1 rounded">Backend Response (truncated)</code>
            </div>
            <pre className="text-xs md:text-sm overflow-x-auto bg-slate-950/60 rounded-xl p-4 border border-slate-800/60 whitespace-pre-wrap">{raw ? JSON.stringify(raw, null, 2) : '—'}</pre>
          </div>
        </div>
      </div>

      <footer className="max-w-7xl mx-auto mt-8 text-slate-400 text-xs">
        <p>Connected to backend: {BACKEND_URL}</p>
      </footer>
    </div>
  )
}
