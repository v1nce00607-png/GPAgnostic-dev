from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os, uuid, json
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import Field
from .models import Strict, save
from .geometry_config import GeometryConfig as PlannerConfig
from .ur_hardware import HardwareSpec as GripperConfig, CADGripper

app = FastAPI(title="Semantic Grasp Planner", version="3.0.2")
pool = ThreadPoolExecutor(max_workers=1)


class PlanRequest(Strict):
    dataset_id: str = Field("0000", pattern=r"^\d{4}$")
    planner: PlannerConfig = Field(default_factory=PlannerConfig)
    grippers: list[GripperConfig] = Field(min_length=1, max_length=6)


def roots():
    return (
        Path(os.environ.get("GRASP_DATA_ROOT", "data")).resolve(),
        Path(os.environ.get("GRASP_RUNS_ROOT", "runs")).resolve(),
    )


def jobpath(job):
    try:
        uuid.UUID(job)
    except ValueError:
        raise HTTPException(400, "Invalid job identifier")
    return roots()[1] / job


@app.get("/health")
def health():
    return {"version": "3.0.2", "status": "ready", "robot_control": False}


@app.get("/v3/schemas")
def schemas():
    return {
        "planner": PlannerConfig.model_json_schema(),
        "gripper": GripperConfig.model_json_schema(),
        "request": PlanRequest.model_json_schema(),
    }


@app.post("/v3/grippers/validate")
def validate(g: GripperConfig):
    return {
        "valid": True,
        "cad_evidence": CADGripper(g).evidence(),
        "specification": g.model_dump(),
        "scope": "configuration validation; not physical verification",
    }


def worker(job, request, data):
    from .pipeline import run

    def progress(msg):
        save(
            job / "job.json", {"job_id": job.name, "status": "running", "progress": msg}
        )

    try:
        progress("Starting")
        save(
            job / "planner.json",
            request.planner.model_copy(
                update={"dataset_id": request.dataset_id}
            ).model_dump(),
        )
        files = []
        for i, g in enumerate(request.grippers):
            f = job / f"gripper_{i}.json"
            save(f, g.model_dump())
            files.append(f)
        result = run(
            data,
            job / "planner.json",
            files,
            job / "results",
            progress,
        )
        save(
            job / "job.json",
            {"job_id": job.name, "status": "completed", "summary": result},
        )
    except Exception as e:
        save(
            job / "job.json",
            {
                "job_id": job.name,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
            },
        )


@app.post("/v3/plan", status_code=202)
def submit(request: PlanRequest):
    data, runs = roots()
    path = (data / request.dataset_id).resolve()
    if not path.is_relative_to(data) or not (path / "camera.json").is_file():
        raise HTTPException(404, "Dataset not installed")
    if len({g.name for g in request.grippers}) != len(request.grippers):
        raise HTTPException(422, "Duplicate gripper names")
    job = runs / str(uuid.uuid4())
    job.mkdir(parents=True)
    save(job / "request.json", request.model_dump())
    save(job / "job.json", {"job_id": job.name, "status": "queued"})
    pool.submit(worker, job, request, path)
    return {
        "job_id": job.name,
        "status": "queued",
        "status_url": f"/v3/jobs/{job.name}",
    }


@app.get("/v3/jobs/{job_id}")
def status(job_id: str):
    p = jobpath(job_id) / "job.json"
    if not p.exists():
        raise HTTPException(404, "Job not found")
    return json.loads(p.read_text())


@app.get("/v3/jobs/{job_id}/files/{name:path}")
def result_file(job_id: str, name: str):
    root = (jobpath(job_id) / "results").resolve()
    f = (root / name).resolve()
    if (
        not f.is_relative_to(root)
        or not f.is_file()
        or f.suffix
        not in {".json", ".csv", ".png", ".svg", ".html", ".js", ".npz", ".ply", ".glb"}
    ):
        raise HTTPException(404, "Artifact not found")
    return FileResponse(f)
