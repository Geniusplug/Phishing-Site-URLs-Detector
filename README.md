# RACE-Phish URL Analyzer

A small FastAPI demo that analyzes a URL locally with a bundled machine-learning model. It does not fetch or execute the submitted website.

## Run locally

Install Python 3.11 or 3.12 and open PowerShell in the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Start the web app:

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

The app loads `models/deployment_url_model.joblib` and `models/deployment_features.json` at startup. Predictions are logged locally to `logs/prediction_logs.sqlite3`; only a truncated SHA-256 URL hash is retained by default rather than the raw URL.

## Deploy from GitHub

GitHub Pages cannot run this FastAPI backend. Connect this repository to Render and deploy the included `render.yaml` blueprint. Render will install `requirements.txt` and start the API on its public port. The live demo URL will then serve both the website and `/api/analyze` from the same service.

## Scope and privacy

The deployment model uses only lexical URL features. It does not query DNS, WHOIS, page content, traffic, or external reputation services. A raw URL is never stored by the app; only a truncated hash and prediction metadata are logged.

Research datasets, training code, paper assets, experiment results, and local logs are intentionally excluded from the GitHub demo repository.
