from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
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
    return render(request, "landing.html")


@app.get("/optimize", response_class=HTMLResponse)
def optimize(request: Request) -> HTMLResponse:
    return render(request, "index.html")


@app.post("/run")
async def run_analysis(
    request: Request,
    requests_file: UploadFile = File(...),
    resources_file: UploadFile = File(...),
    load_balancer_cost: float = Form(...),
    epochs: int = Form(5),
) -> StreamingResponse:
    async def stream_results():
        temp_paths: list[Path] = []
        progress_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def report_progress(stage: str, message: str) -> None:
            loop.call_soon_threadsafe(
                progress_queue.put_nowait,
                {"type": "progress", "stage": stage, "message": message},
            )

        def encode(event: dict[str, object]) -> str:
            return json.dumps(event) + "\n"

        yield encode({"type": "progress", "stage": "queued", "message": "Preparing your analysis"})
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

            task = asyncio.create_task(
                asyncio.to_thread(
                    run_pipeline_report,
                    str(temp_paths[0]),
                    str(temp_paths[1]),
                    load_balancer_cost,
                    epochs,
                    report_progress,
                )
            )
            while not task.done() or not progress_queue.empty():
                try:
                    event = await asyncio.wait_for(progress_queue.get(), timeout=0.25)
                    yield encode(event)
                except asyncio.TimeoutError:
                    continue

            report = await task
            yield encode({
                "type": "complete",
                "html": templates.get_template("report_fragment.html").render(report=report),
            })
        except Exception as exc:
            yield encode({"type": "error", "message": str(exc)})
        finally:
            for path in temp_paths:
                path.unlink(missing_ok=True)

    return StreamingResponse(
        stream_results(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache"},
    )
