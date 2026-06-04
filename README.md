# No-Code Data Cleaning & Automated ML Platform

FastAPI backend + React (Vite) frontend. Upload a CSV, configure target/imputation, run AutoML (Logistic Regression, Random Forest, XGBoost + Optuna), download the trained sklearn `Pipeline` as `.pkl`.

## Local development

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite proxies `/api` to `http://127.0.0.1:8000`. Open http://localhost:5173

## Deploy

| Layer    | Platform | Notes |
|----------|----------|--------|
| Frontend | Vercel   | Set `VITE_API_URL` to your Render API URL (e.g. `https://nocode-ml-api.onrender.com`) |
| Backend  | Render   | Use `render.yaml` or set start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT` with root `backend` |

On Render, set `FRONTEND_ORIGIN` to your Vercel URL for CORS.

## API

- `POST /api/analyze-csv` — column stats + 5-row preview
- `POST /api/process-and-train` — form: `file`, `target_column`, `imputation_strategy` (`mean_median` \| `drop`)
- `GET /api/download-model/{model_id}` — download `.pkl` pipeline
- `POST /api/download-cleaned-csv` — form: `file`, `target_column`, `imputation_strategy` → cleaned CSV
