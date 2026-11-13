# Capacity / Forecast

A FastAPI interface for forecasting hourly request demand and selecting a least-cost resource configuration.

## Requirements

- Python 3.10+
- TensorFlow, pandas, NumPy, and scikit-learn

Install the dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the web app

```powershell
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> in a browser.

## Deploy on Render

This repository includes a [`render.yaml`](./render.yaml) Blueprint. In Render, choose **New > Blueprint**, connect this repository, and apply the Blueprint. Render will install the dependencies, start the FastAPI service on its assigned port, and use `/health` for health checks.

The service uses Python 3.11.11 from [`runtime.txt`](./runtime.txt), which is compatible with the TensorFlow dependency. No environment variables or secrets are required for the current application.

## Input files

The upload form expects two CSV files:

### Request history

A `Requests` column with at least 336 hourly observations. Values must be finite and non-negative.

```csv
Requests
1200
1350
...
```

### Instance catalogue

The columns `name`, `capacity_rps`, and `cost_per_hour` are required.

```csv
name,capacity_rps,cost_per_hour
small,25,0.01
large,100,0.04
```

The report includes the one-week forecast peak, required capacity, and a comparison of every optimization method:

- **Greedy**: cost-per-capacity heuristic baseline
- **Genetic**: evolutionary baseline without local search
- **Exact**: dynamic-programming least-cost reference
- **Memetic**: evolutionary search with repair and local search

Each method displays its selected configuration, hourly cost, optimality gap against the exact result, and runtime. Pipeline timing for training, forecasting, and optimization is also shown.

## CLI usage

The original pipeline can also be run directly:

```powershell
python resource_optimization_pipeline.py --requests requests_every_hour.csv --resources instance_types.csv --load-balancer-cost 0.025 --epochs 5
```
