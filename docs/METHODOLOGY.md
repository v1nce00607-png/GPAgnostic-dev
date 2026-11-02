# Research methodology

## Research question
Can an object-aware, geometry-driven pipeline generate valid grasps for multiple two-finger gripper geometries without a separate hand-designed planner for each gripper?

## Ten gates
1. `data_alignment`
2. `hardware_geometry`
3. `kinematics`
4. `contact_geometry`
5. `self_collision`
6. `target_body`
7. `environment`
8. `approach`
9. `closure`
10. `filter_policy`

Only a candidate satisfying **all required checks under the same documented model assumptions** may be designated Full Geometric PASS.

## Distinctions to preserve
- A point-cloud semantic confidence score is not physical ground truth.
- An Environment PASS is not a Full Geometric PASS.
- Multiple ranks/tilts for the same contact pair are not independent grasp locations.
- Non-watertight target CAD poses represent collision uncertainty; local pad-contact exemptions require explicit model-assumption reporting.
- Removed background points are retained as auditable uncertainty evidence; aggressive ROI erasure must be reported as an ablation, not scene certification.
- Simulated geometric feasibility is not physical stability or executed robot motion.

## Evaluation protocol
Keep capture frames, calibration, CAD revision, seed, software revision, selected gripper geometry and thresholds fixed for controlled comparisons. Compare result populations by gate and by target/gripper. Do not treat rank indices in candidate IDs as stable pose IDs across runs. Archive manifest and SHA-256 digests for large external results.
