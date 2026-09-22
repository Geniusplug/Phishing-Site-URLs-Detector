# RACE-Phish: Research Repository and URL Analyzer

This repository contains the reproducible research code, dataset, trained models, evaluation tables, and FastAPI demonstration for the RACE-Phish phishing URL study. The demo does not fetch or execute submitted websites.

## Reproduce the research run

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

Install the full research environment with `requirements.txt`, then run:

```powershell
python -m src.pipeline 100
```

This trains and evaluates the baselines and RACE-Phish model, runs ablations and robustness analysis, and writes tables and figures under `results/`.

## Run the demo locally

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000

The app loads `models/deployment_url_model.joblib` and `models/deployment_features.json` at startup. Predictions are logged locally to `logs/prediction_logs.sqlite3`; only a truncated SHA-256 URL hash is retained by default rather than the raw URL.

## Deploy from GitHub

GitHub Pages cannot run this FastAPI backend. Connect this repository to Render and deploy the included `render.yaml` blueprint. Render installs the smaller `requirements-deployment.txt` and starts the API on its public port. The live demo URL will then serve both the website and `/api/analyze` from the same service.

## Scope and privacy

The deployment model uses only lexical URL features. It does not query DNS, WHOIS, page content, traffic, or external reputation services. A raw URL is never stored by the app; only a truncated hash and prediction metadata are logged.

Research datasets, training code, paper assets, experiment results, and the deployment demo are included for reproducibility. Local logs, Python caches, and optional training checkpoints are excluded.
