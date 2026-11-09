import argparse, json, os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Grasp Planner V3.0 research pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("--dataset", required=True)
    a.add_argument("--config", required=True)
    a.add_argument("--gripper", action="append", required=True)
    a.add_argument("--output", required=True)
    a = sub.add_parser("serve")
    a.add_argument("--data-root", required=True)
    a.add_argument("--runs-root", required=True)
    a.add_argument("--port", type=int, default=8000)
    a = sub.add_parser("schemas")
    a.add_argument("--output", default="docs")
    args = p.parse_args()
    if args.cmd == "run":
        from .pipeline import run

        print(
            json.dumps(
                run(
                    args.dataset,
                    args.config,
                    args.gripper,
                    args.output,
                ),
                indent=2,
            )
        )
    elif args.cmd == "serve":
        import uvicorn

        os.environ["GRASP_DATA_ROOT"] = str(Path(args.data_root).resolve())
        os.environ["GRASP_RUNS_ROOT"] = str(Path(args.runs_root).resolve())
        uvicorn.run("grasp_planner.api:app", host="127.0.0.1", port=args.port)
    else:
        from .models import save
        from .geometry_config import GeometryConfig as PlannerConfig
        from .ur_hardware import HardwareSpec as GripperConfig

        for c in [PlannerConfig, GripperConfig]:
            save(
                Path(args.output) / (c.__name__ + ".schema.json"), c.model_json_schema()
            )


if __name__ == "__main__":
    main()
