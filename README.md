# VEDAS NDVI → Yield (Monorepo)

Monorepo with **FastAPI backend** and **React (Vite) frontend** that pulls NDVI time series from VEDAS (with an API key) and predicts yield using your `final.py` predictor.

## Structure
```
vedas-app/
  backend/
    app.py
    vedas_client.py
    predictor_wrapper.py
    final.py                 # (copied from your local)
    requirements.txt
    Dockerfile
    .env.example
  frontend/
    package.json
    vite.config.js
    index.html
    src/
      main.jsx
      App.jsx
render.yaml                  # Render blueprint for one-click deploy
```

## Local Setup

### Backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # put your VEDAS_API_KEY here
uvicorn app:app --reload
```
Docs at: http://localhost:8000/docs

### Frontend (Vite + React)
```bash
cd frontend
npm install
npm run dev
```
Open http://localhost:5173 and set the **Backend URL** to `http://localhost:8000` (or your Render backend URL).

## Deploy on Render (Blueprint)
1. Push this repo to GitHub.
2. On Render, click **New > Blueprint** and point to your GitHub repo.
3. Render reads `render.yaml` and creates:
   - A **Web Service** for the FastAPI backend
   - A **Static Site** for the React frontend
4. Add `VEDAS_API_KEY` in the backend service’s environment variables.
5. Deploy.

## Manual Deploy (without blueprint)
- Create a Render **Web Service** with root directory `backend`, build command `pip install -r requirements.txt`, start `uvicorn app:app --host 0.0.0.0 --port $PORT`.
- Create a Render **Static Site** with root directory `frontend`, build `npm install && npm run build`, publish directory `dist`.

## Security
- Never commit `.env`. Keep your VEDAS API key in Render env variables.
- Update `vedas_client.py` with the exact VEDAS endpoint your account provides.

## Notes
- `final.py` is included from your local upload. Replace it with your latest predictor if needed.
- If VEDAS returns extra fields (LST/rainfall/soil moisture), map them in `predictor_wrapper.py` for better accuracy.
