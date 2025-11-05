import React, { useState } from 'react'
import axios from 'axios'
import { Line } from 'react-chartjs-2'
import 'chart.js/auto'

export default function App() {
  const [lat, setLat] = useState('')
  const [lon, setLon] = useState('')
  const [backendUrl, setBackendUrl] = useState('') // set your Render backend URL here
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const submit = async (e) => {
    e.preventDefault()
    setLoading(true)
    try {
      const url = (backendUrl || '').trim()
      const endpoint = url ? `${url.replace(/\/$/,'')}/api/predict` : '/api/predict'
      const resp = await axios.post(endpoint, { lat: parseFloat(lat), lon: parseFloat(lon) })
      setResult(resp.data)
    } catch (err) {
      alert('Error: ' + (err.response?.data?.detail || err.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{padding: 24, fontFamily: 'Inter, system-ui, Arial'}}>
      <h1>VEDAS NDVI → Yield</h1>
      <p>Enter coordinates. Optionally set your deployed backend URL (Render) for production.</p>
      <form onSubmit={submit} style={{display:'flex', gap:12, flexWrap:'wrap', alignItems:'center'}}>
        <label>Latitude <input value={lat} onChange={e=>setLat(e.target.value)} required/></label>
        <label>Longitude <input value={lon} onChange={e=>setLon(e.target.value)} required/></label>
        <label>Backend URL <input placeholder="https://your-backend.onrender.com" value={backendUrl} onChange={e=>setBackendUrl(e.target.value)} style={{minWidth:320}}/></label>
        <button type="submit" disabled={loading}>{loading ? 'Working…' : 'Predict'}</button>
      </form>

      {result && (
        <div style={{marginTop: 24}}>
          <h2>Prediction</h2>
          <p><b>Predicted yield:</b> {Math.round(result.predicted_yield)} kg/ha</p>
          <p><b>Average NDVI:</b> {result.avg_ndvi?.toFixed?.(3)}</p>
          <p><b>Samples used:</b> {result.meta?.samples_used}</p>

          <h3>NDVI Time Series</h3>
          <Line data={{
            labels: (result.ndvi_timeseries || []).map(d => d.date),
            datasets: [{ label: 'NDVI', data: (result.ndvi_timeseries || []).map(d => d.ndvi) }]
          }}/>
        </div>
      )}
    </div>
  )
}
