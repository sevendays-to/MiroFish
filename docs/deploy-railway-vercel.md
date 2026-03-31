# Deploying MiroFish with Railway + Vercel

This setup keeps the backend on Railway and the frontend on Vercel:

- Railway runs the Flask backend from `backend/` using Railpack and Gunicorn
- Vercel serves the Vue/Vite app from `frontend/`
- The frontend calls the Railway backend directly via `VITE_API_BASE_URL`

## Backend on Railway

1. Create a Railway service from this repository.
2. Set the Railway root directory to `/backend`.
3. Railway will read [`backend/railway.json`](../backend/railway.json), which forces Railpack and starts Gunicorn with `wsgi:app`.
4. Add a persistent volume mounted at `/app/backend/uploads`.
5. Set these Railway variables:

```env
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL_NAME=gpt-4o-mini
ZEP_API_KEY=...
SECRET_KEY=replace-me
FLASK_DEBUG=false
CORS_ORIGINS=https://your-frontend.vercel.app
```

6. Generate a Railway public domain for the backend service.

Notes:

- Python is pinned via [`backend/.python-version`](../backend/.python-version) to Python 3.11 because `camel-oasis` requires `<3.12`.
- Production boot uses Gunicorn via [`backend/wsgi.py`](../backend/wsgi.py) and [`backend/gunicorn.conf.py`](../backend/gunicorn.conf.py).
- There is intentionally no `backend/Dockerfile` anymore. If Railway still shows Docker, the service is deploying an older commit or the wrong branch/root directory.
- The backend is intentionally fixed to a single Gunicorn worker. Do not enable multi-worker or horizontal scaling for this v1 deployment because task state and simulations rely on local files and in-memory state.
- Deploys or restarts can interrupt active simulations. Schedule deploys outside active runs.

## Frontend on Vercel

1. Create a Vercel project from this repository.
2. Set the project root directory to `frontend`.
3. Add:

```env
VITE_API_BASE_URL=https://your-backend.up.railway.app
```

4. Deploy.

Notes:

- [`frontend/vercel.json`](../frontend/vercel.json) enables SPA route fallback for Vue Router history mode.
- Production builds require `VITE_API_BASE_URL`. Local development still falls back to `http://localhost:5001`.

## Smoke checks

After both deploys are live:

1. Open `https://your-backend.up.railway.app/health` and confirm a `200` response.
2. Open the Vercel frontend and verify there are no browser CORS errors.
3. Refresh a non-root frontend route to confirm the SPA fallback works.
4. Run a minimal workflow: upload a document, build a graph, create a simulation, start it, and generate a report.
5. Restart the Railway service and confirm data under `/app/backend/uploads` persists.
