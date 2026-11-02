# Architecture

The existing project package is intentionally preserved as `grasp_planner/` to keep imports and CLI working.

- **Input / data model:** `models.py` describes captures and annotated objects; `cli.py` exposes `run` / `serve` / `schemas`.
- **Reconstruction:** `reconstruction.py` fuses semantic RGB-D observations into a point cloud.
- **Cleaning:** `cleaning.py` audits statistical outliers, depth evidence and object-relative occupancy assumptions.
- **Hardware:** `ur_hardware.py` loads gripper CAD/URDF, joint limits, fingertip poses and kinematic parameters.
- **Planning:** `geometry_planner.py` proposes opposing CAD contacts, positions the gripper and ranks candidates.
- **Collision:** `geometry_collision.py` checks target/object/scene obstacles and swept motions under explicit assumptions.
- **Trajectories:** `trajectory.py` produces time-parameterized gripper paths; robot arm IK is out of scope.
- **Orchestration:** `pipeline.py` manages end-to-end processing and output.
- **Reports:** `geometry_report.py` writes geometric diagnostics and offline visualization assets.

Keep plugin/gripper adapters separate from the geometric reasoning. A new gripper must supply geometry, kinematics and correctly calibrated contact dimensions; supporting arbitrary end-effectors is an open research goal.
