import json
import sys
from pathlib import Path
import yaml

from grasp_planner.models import Capture
from grasp_planner.pipeline import load_reconstruction
from grasp_planner.geometry_config import GeometryConfig
from grasp_planner.ur_hardware import HardwareSpec
from grasp_planner.geometry_report import render

def main():
    dataset, out, config, *grippers = map(Path, sys.argv[1:])

    if len(grippers) != 3:
        raise ValueError("Three gripper configurations required")

    cfg = GeometryConfig.model_validate(
        yaml.safe_load(config.read_text(encoding="utf-8"))
    )

    specs = [
        HardwareSpec.model_validate(
            yaml.safe_load(path.read_text(encoding="utf-8"))
        )
        for path in grippers
    ]

    cap = Capture(dataset)
    rec = load_reconstruction(out)

    rows = json.loads(
        (out / "all_candidates.json").read_text(encoding="utf-8")
    )

    print(
        f"Resuming report only: {len(rows)} candidates",
        flush=True
    )

    render(out, cap, rec, rows, specs, cfg)

    print(
        f"Report completed: {out / 'report.html'}",
        flush=True
    )

if __name__ == "__main__":
    main()