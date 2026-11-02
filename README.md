# GPAgnostic — Gripper-Agnostic Geometric Grasp Planning

**Research prototype | RGB-D Reconstruction | Geometric Collision Validation | Robotics**

## Overview

GPAgnostic is a research project focused on developing a gripper-agnostic grasp planning framework using semantically annotated 3D reconstructions.

The system combines RGB-D observations, object CAD models, and user-defined gripper geometry to automatically generate and evaluate grasp candidates across different robotic grippers.

The primary objective is to reduce gripper-specific manual configuration while preserving transparent and reproducible geometric validation.

## Research Objectives

- Develop a common grasp-planning pipeline for different two-finger robotic grippers.
- Generate candidate grasps from reconstructed object geometry.
- Detect gripper-object, gripper-environment, and self-collisions.
- Evaluate approach and finger-closure trajectories.
- Improve reconstruction filtering while preserving collision-relevant observations.
- Compare algorithmic improvements using reproducible experiments.

## Supported Grippers

| Gripper | Model |
|---|---|
| OnRobot RG2 | Two-finger gripper |
| OnRobot RG6 | Two-finger gripper |
| OnRobot 2FG7 | Parallel gripper |

Gripper geometry and kinematics are derived from CAD/URDF descriptions.

## Research Dataset

The primary experiment uses 318 RGB-D frames with four annotated object instances:

- Pliers_001
- Pliers_002
- Rachet
- Screwdriver

The pipeline integrates semantic masks, camera poses, object geometry, and observed scene occupancy.

## Methodology

The planning framework consists of:

1. Multi-view RGB-D reconstruction
2. Semantic and geometric point-cloud filtering
3. Object-aware scene reconstruction
4. Gripper geometry and kinematics integration
5. Contact-based grasp candidate generation
6. Geometric candidate ranking
7. Target and environment collision validation
8. Approach and closure trajectory evaluation
9. Result visualization and experiment reporting

The current research focuses on resolving target-body collision uncertainty, improving grasp placement, and distinguishing valid observations from reconstruction artifacts.

## Current Limitations

The current implementation evaluates geometric feasibility only.

Robot-arm inverse kinematics, force closure, dynamic stability, and physical robotic execution are outside the present validation scope.

Some target CAD meshes are non-watertight, which introduces uncertainty in solid collision checking. Aggressive occupancy filtering is treated as an experimental scene assumption rather than proven empty-space reconstruction.

## Future Development

- Improved target contact and body-collision modeling
- More robust occupancy reconstruction
- Better grasp candidate diversity and ranking
- Increased jointly collision-free grasp candidates
- Cross-object and cross-gripper evaluation
- Reproducible geometric benchmarks

## Project Status

**Active research and development.**

The repository preserves algorithm versions, configurations, experimental results, and validation limitations to support further scientific evaluation.
