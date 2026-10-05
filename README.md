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

The report includes the one-week forecast peak, required capacity, greedy baseline, and memetic optimization result.

## CLI usage

The original pipeline can also be run directly:

```powershell
python resource_optimization_pipeline.py --requests requests_every_hour.csv --resources instance_types.csv --load-balancer-cost 0.025 --epochs 5
```
