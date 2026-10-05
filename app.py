from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from resource_optimization_pipeline import run_pipeline_report


BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Capacity / Forecast")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


def render(request: Request, template: str, **context: object) -> HTMLResponse:
    return templates.TemplateResponse(request=request, name=template, context=context)


@app.get("/", response_class=HTMLResponse)
def home(request: Request) -> HTMLResponse:
    return render(request, "index.html")


@app.post("/run", response_class=HTMLResponse)
async def run_analysis(
    request: Request,
    requests_file: UploadFile = File(...),
    resources_file: UploadFile = File(...),
    load_balancer_cost: float = Form(...),
    epochs: int = Form(5),
) -> HTMLResponse:
    temp_paths: list[Path] = []
    try:
        if not requests_file.filename or not resources_file.filename:
            raise ValueError("Choose both CSV files before running the analysis.")
        if load_balancer_cost < 0:
            raise ValueError("Load balancer cost cannot be negative.")
        if epochs < 1 or epochs > 100:
            raise ValueError("Epochs must be between 1 and 100.")

        for upload in (requests_file, resources_file):
            suffix = Path(upload.filename or ".csv").suffix.lower()
            if suffix != ".csv":
                raise ValueError("Both uploads must be CSV files.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(await upload.read())
                temp_paths.append(Path(temporary.name))

        report = run_pipeline_report(
            str(temp_paths[0]),
            str(temp_paths[1]),
            load_balancer_cost,
            epochs,
        )
        return render(request, "results.html", report=report)
    except Exception as exc:
        return render(request, "error.html", error=str(exc))
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
