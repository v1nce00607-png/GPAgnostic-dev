# Grasp Planner: Geometric Grasp Planning in Semantically Annotated 3D Scenes

This research prototype generates and geometrically validates grasp candidates using an RGB-D-based 3D reconstruction, object CAD models, and manufacturer-provided gripper models. I evaluated the system with three OnRobot parallel-jaw grippers: RG2, RG6, and 2FG7.

## Research Scope and Limitations

A `PASS` result means that a grasp candidate satisfies the configured geometric checks. It does **not** demonstrate physical grasp stability, robot-arm inverse kinematics feasibility, or successful execution on a real robot. Some candidates marked `PASS` do not have independently verified bilateral contact between the original equipment manufacturer (OEM) finger pads and the object. Object-adjacent point removal during point-cloud filtering also means that geometric decisions must be interpreted with caution.

## Repository Structure

```text
grasp_planner/             Core algorithms and gripper adapters
  assets/                 OnRobot manufacturer models and license
configs/                  Input and gripper configurations
scripts/                  Import, auditing, benchmarking, and reporting tools
tests/                    Automated tests
docs/                     Architecture, data schema, results, and API documentation
examples/                 Example client
data/                     Local input data (excluded by .gitignore)
results/                  Local results (excluded by .gitignore)
provenance/               Data provenance and checksums
```

## Installation

Python 3.10 or newer is required, along with a platform-compatible installation of `python-fcl`.

```powershell
Set-Location 'C:\ProjektGP\GPagnostic-dev'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q tests
```

The following example runs the planner with the previously prepared dataset. Adjust the input and output paths to match your local environment:

```powershell
Set-Location 'C:\ProjektGP\GPagnostic-dev'
.\.venv\Scripts\python.exe -m grasp_planner.cli run --dataset 'C:\ProjektGP\Hand_tools\0000' --config '.\configs\planner_geometry_final.yaml' --gripper '.\configs\grippers\2fg7.yaml' --gripper '.\configs\grippers\rg2.yaml' --gripper '.\configs\grippers\rg6.yaml' --output 'C:\ProjektGP\GPagnostic_run_new'
```

Some configurations are computationally demanding, and the planner requires the original capture data. The automated tests and planner were not rerun when this GitHub-ready package was prepared.

## Documentation

- `docs/ARCHITECTURE.md`: Data flow, modules, and important functions.
- `docs/RESULTS_SUMMARY.md`: Results and their interpretation, including reproducibility considerations.
- `docs/METHOD_AND_TDK.md`: Detailed technical background for the research project.
- `NOTICE.md`: Data provenance and information about third-party materials.

This repository was prepared as a local project and was not automatically published to GitHub. Before making it public, verify the licensing of your own code and the usage terms of all included manufacturer-provided data.
