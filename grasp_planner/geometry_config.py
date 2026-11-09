from pydantic import Field,model_validator
from typing import Literal
from .models import Strict
class GeometryConfig(Strict):
    dataset_id: str='0000'
    seed: int=7301
    reference_frame: int=Field(0,ge=0)
    pixel_stride: int=Field(4,ge=1,le=32)
    frame_stride: int=Field(1,ge=1)
    voxel_m: float=Field(.003,gt=0,le=.01)
    adaptive_voxel: bool=True
    roi_margin_m: float=Field(.22,gt=0)
    depth_min_m: float=Field(.15,gt=0)
    depth_max_m: float=Field(2.,gt=0)
    depth_jump_m: float=Field(.025,gt=0)
    mask_erosion_px: int=Field(1,ge=0,le=3)
    semantic_confidence: float=Field(.8,ge=.5,le=1)
    min_voxel_observations: int=Field(2,ge=1)
    sensor_uncertainty_m: float=Field(.004,ge=0)
    environment_clearance_m: float=Field(.002,ge=0)
    # OPT7: an environment (observed-occupancy) collision needs at least this many DIFFERENT occupied voxels
    # inside the inflated link volume; fewer are treated as isolated floating points and ignored.
    # 1 = original behaviour (a single point rejects the pose).
    environment_min_occupied_voxels: int=Field(1,ge=1,le=500)
    # OPT7: background (label 0) points within this distance of the TARGET object's own CAD surface are treated as part of
    # the target (rim / shadow / edge points of its own segmentation), not as obstacles. A pad touching the target is by
    # definition within a few mm of its surface. 0 = off (original behaviour). Alignment uncertainty: median 1.7, p95 5.5 mm.
    environment_ignore_target_shell_m: float=Field(0.,ge=0.,le=.02)
    # OPT7: background (label 0) points at most this far ABOVE the fitted table plane are table-surface noise (plane RMS ~2.9 mm,
    # 3 sigma ~8.6 mm) and are already covered by the support-plane check, so they are not obstacles. 0 = off.
    environment_ignore_table_band_m: float=Field(0.,ge=0.,le=.015)
    candidate_height_quantiles: list[float]=Field(default_factory=lambda:[.2,.4,.6,.78,.9,.98])
    target_penetration_tolerance_m: float=Field(.0005,ge=0,le=.001)
    alignment_median_limit_m: float=Field(.008,gt=0)
    alignment_p95_limit_m: float=Field(.025,gt=0)
    sor_k: int=Field(20,ge=5,le=100)
    sor_std: float=Field(2.,gt=0)
    radius_m: float=Field(.009,gt=0)
    radius_min_neighbors: int=Field(5,ge=1)
    dbscan_eps_m: float=Field(.01,gt=0)
    dbscan_min_samples: int=Field(8,ge=2)
    dbscan_min_cluster: int=Field(50,ge=5)
    occupancy_min_views: int=Field(3,ge=1)
    filter_min_views: int=Field(3,ge=2)
    evidence_frames: int=Field(64,ge=5,le=318)
    bbox_scale: float=Field(1.25,ge=1.,le=1.5)
    bbox_policy: Literal['evidence','erase_unverified']='evidence'
    relabel_band_m: float=Field(.008,ge=0,le=.01)
    relabel_min_views: int=Field(3,ge=2)
    relabel_vote_fraction: float=Field(.6,ge=.5,le=1)
    # An explicit opt-in: accept a small minority of competing mask votes only
    # when the target evidence dominates. Default 0 reproduces baseline.
    relabel_max_competitor_fraction: float=Field(0.,ge=0.,le=.2)
    relabel_min_dominance_ratio: float=Field(5.,ge=3.,le=100.)
    # Select promising candidates using the SAME collision gates as validation.
    # 1 = historical selector; 2 = inspect twice as many candidates, report
    # screening cost and keep the original final candidate count.
    candidate_screen_multiplier: int=Field(1,ge=1,le=5)
    # OPT3: no screen-space "above" assumption. Require independent 3D
    # isolation AND multiple calibrated depth rays proving free space.
    floating_policy: Literal['off','depth_confirmed']='off'
    floating_min_height_m: float=Field(.012,ge=.008,le=.1)
    floating_min_free_views: int=Field(8,ge=5,le=64)
    floating_free_to_occupied_ratio: float=Field(4.,ge=3.,le=20.)
    # A contact is only exempted from the convex target overlap if ALL FCL
    # witnesses are confined to an OEM fingertip contact patch and shallow.
    target_contact_policy: Literal['convex_strict','local_pad_witness']='convex_strict'
    target_contact_patch_radius_m: float=Field(.0025,gt=0,le=.004)
    # 'fixed': shallow contact allowed within target_contact_patch_radius_m of an anchor (default).
    # 'pad_face_rect': shallow contact (depth <= target_penetration_tolerance_m) allowed anywhere on the
    # OEM pad's inner face rectangle (width x length) enlarged by target_contact_patch_margin_m on every side.
    target_contact_patch_mode: Literal['fixed','pad_face_rect']='fixed'
    target_contact_patch_margin_m: float=Field(.0005,ge=0,le=.003)
    target_contact_surface_distance_m: float=Field(.0006,ge=0,le=.001)
    approach_precheck_for_joint_pass: bool=False
    target_body_clearance_ranking: bool=False
    # OPT4 ablation: spatially bounded background suppression around target CAD.
    # "experimental_roi" is NOT a free-space certificate. Every omitted point
    # is recorded in filter_evidence.npz and compared to the unmodified OPT3.
    halo_policy: Literal['off','isolated_only','experimental_roi']='off'
    halo_objects: list[str]|None=None
    halo_margin_m: float=Field(.018,ge=.003,le=.05)
    halo_ceiling_margin_m: float=Field(.035,ge=.005,le=.08)
    halo_plane_guard_m: float=Field(.008,ge=.005,le=.025)
    halo_other_object_guard_m: float=Field(.006,ge=.002,le=.02)
    # OPT5: joint feasibility search (same geometry checks, no relaxed clearance).
    opt5_enabled: bool=False
    opt5_refine_seeds_per_group: int=Field(6,ge=0,le=20)
    opt5_refine_roll_deg: list[float]=Field(default_factory=lambda:[-45.,-25.,-12.,12.,25.,45.])
    opt5_approach_lateral_m: list[float]=Field(default_factory=lambda:[.025,.050])
    opt5_closure_step_m: float=Field(.002,gt=0,le=.01)
    # OPT5 adaptive CAD-contact experiment. Nonwatertight CAD cannot certify a solid.
    # original_surface_ablation allows conditional surface-only tests and explicitly
    # reports potential convex-hull overlaps; conservative_hull preserves baseline.
    target_body_model: Literal['conservative_hull','original_surface_ablation','repaired_solid','convex_decomposition']='conservative_hull'
    # OPT7 CAD repair / convex decomposition of the TARGET model (used only by
    # target_body_model in {'repaired_solid','convex_decomposition'}).
    cad_repair_pitch_m: float=Field(.0005,ge=.0002,le=.002)
    cad_repair_max_faces: int=Field(40000,ge=5000,le=200000)
    cad_decomp_threshold: float=Field(.03,ge=.01,le=.2)
    cad_decomp_inflate_m: float=Field(0.,ge=0.,le=.002)
    cad_cache_dir: str|None=None
    # OPT7: collision model of the GRIPPER links. 'oem_as_is' keeps the original behaviour
    # (non-watertight links -> convex cover). 'repaired_solid' reconstructs each non-watertight
    # OEM link as an audited watertight solid for the static target/environment checks;
    # sweeps (approach/closure) stay convex covers, self-collision stays on original triangles.
    # OPT7 step 1: reuse FCL geometry per link instead of rebuilding it every query (same verdicts, faster).
    fcl_persistent_objects: bool=True
    gripper_collision_model: Literal['oem_as_is','repaired_solid']='oem_as_is'
    gripper_repair_pitch_m: float=Field(.0003,ge=.0002,le=.002)
    gripper_collision_max_faces: int=Field(20000,ge=2000,le=100000)
    adaptive_closure_enabled: bool=False
    adaptive_obb_half_window_m: float=Field(.012,gt=.003,le=.05)
    adaptive_obb_side_clearance_m: float=Field(.003,ge=.0005,le=.02)
    adaptive_contact_tolerance_m: float=Field(.0004,gt=.00005,le=.001)
    adaptive_contact_search_steps: int=Field(17,ge=7,le=49)
    adaptive_contact_undertravel_m: float=Field(.001,ge=0,le=.004)
    adaptive_attempts_per_object: int=Field(48,ge=0,le=200)
    adaptive_mount_x_offsets_m: list[float]=Field(default_factory=lambda:[0.,-.001,.001,-.002,.002])
    # OPT6 bounded CAD contact-goal optimization. Disabled by default:
    # no collision tolerances, target CAD, or occupancy model are modified.
    opt6_goal_optimizer_enabled: bool=False
    opt6_goal_attempts_per_object: int=Field(12,ge=0,le=1000)
    opt6_goal_translation_limit_m: float=Field(.008,ge=.001,le=.012)
    opt6_goal_extra_close_m: float=Field(.012,ge=.002,le=.03)
    opt6_goal_min_closure_m: float=Field(.0005,gt=0,le=.005)
    opt6_goal_multistart: int=Field(3,ge=1,le=5)
    opt6_goal_max_nfev_per_seed: int=Field(35,ge=8,le=100)
    opt6_goal_max_distance_evaluations: int=Field(650,ge=100,le=2500)
    # OPT7 (all OFF by default; no gate, tolerance or occupancy is relaxed).
    # RRT-Connect plans the fixed-orientation OPEN-gripper mount translation with
    # the SAME swept-convex edge validator as the straight/waypoint corridors.
    opt7_rrt_enabled: bool=False
    opt7_rrt_max_iterations: int=Field(300,ge=10,le=5000)
    opt7_rrt_max_edge_checks: int=Field(400,ge=20,le=10000)
    opt7_rrt_step_m: float=Field(.02,ge=.005,le=.05)
    opt7_rrt_lateral_extent_m: float=Field(.12,ge=.02,le=.4)
    opt7_rrt_back_extent_m: float=Field(.25,ge=.05,le=.6)
    opt7_rrt_shortcut_passes: int=Field(20,ge=0,le=200)
    # Report-only diagnostics: 180-degree twin and jaw-axis shift scan. They
    # NEVER change status, pose or ranking of a candidate.
    opt7_diagnostics_enabled: bool=False
    opt7_diagnostic_attempts_per_object: int=Field(24,ge=0,le=200)
    opt7_jaw_shift_mm: list[float]=Field(default_factory=lambda:[-4.,-2.,2.,4.])
    # OPT7 collision-aware contact-goal stage (OFF by default). After the OPT6
    # contact-only search has produced bilateral-contact poses that the static
    # gates reject, a second bounded search moves the mount (X,Y,Z), the OEM
    # aperture and a roll about the jaw axis to REDUCE target penetration and the
    # environment-clearance deficit while keeping both pad contacts. Every result
    # still passes the unmodified Scene.target_clear / Scene.clear / self-collision.
    opt7_goal_collision_aware: bool=False
    opt7_goal_collision_max_evals: int=Field(150,ge=20,le=3000)
    opt7_goal_roll_limit_deg: float=Field(10.,ge=0.,le=20.)
    opt7_goal_env_safety_m: float=Field(.0005,ge=0.,le=.003)
    # Skip the stage for a start whose penetration is already hopeless (saves time).
    opt7_goal_collision_skip_depth_m: float=Field(.006,ge=.001,le=.05)
    # Strong pre-filter for the (expensive) collision-aware stage: a start is only optimized if a cheap probe shows
    # penetration <= skip_depth and clearance deficit <= skip_deficit; at most `budget` optimizations per gripper/object
    # (candidates are processed best-ranked first, so the budget goes to the most promising ones).
    opt7_goal_collision_skip_deficit_m: float|None=Field(None,ge=.0005,le=.05)
    opt7_goal_collision_budget_per_object: int|None=Field(None,ge=1,le=2000)
    opt7_goal_penetration_weight: float=Field(1.,gt=0.,le=20.)
    opt7_goal_clearance_weight: float=Field(1.,gt=0.,le=20.)
    # OPT5 joint-static repair. Searches OPEN pregrasp states only: all
    # reported target/environment PASS values are checked, never inferred.
    # A pregrasp static PASS is NOT a CAD-contact or full-grasp PASS.
    opt5_static_pair_enabled: bool=False
    opt5_static_pair_attempts_per_object: int=Field(24,ge=0,le=200)
    opt5_static_pair_max_checks: int=Field(15,ge=1,le=50)
    # Exact non-convex mesh may replace a hull only if it is watertight.
    # Non-watertight targets always retain conservative convex envelopes.
    prefer_exact_watertight_meshes: bool=False
    depth_agreement_m: float=Field(.008,gt=0)
    plane_threshold_m: float=Field(.0035,gt=0)
    plane_min_points: int=Field(80,ge=30)
    max_obstacle_planes: int=Field(8,ge=0,le=20)
    candidates_per_object: int=Field(200,ge=10,le=1000)
    yaw_count: int=Field(12,ge=4,le=32)
    axial_samples: int=Field(19,ge=5,le=40)
    tilt_degrees: list[float]=Field(default_factory=lambda:[0.,20.,40.])
    max_contact_angle_deg: float=Field(35.,gt=0,le=75)
    approach_distance_m: float=Field(.10,gt=0)
    max_tcp_speed_m_s: float=Field(.08,gt=0)
    max_tcp_accel_m_s2: float=Field(.3,gt=0)
    path_sample_m: float=Field(.005,gt=0)
    closure_step_rad: float=Field(.04,gt=0,le=.1)
    top_k: int=Field(10,ge=1,le=50)
    target_instances: list[str]|None=None
    @model_validator(mode='after')
    def ranges(self):
        if self.depth_min_m>=self.depth_max_m:raise ValueError('Invalid depth interval')
        if any(abs(t)>70 for t in self.tilt_degrees):raise ValueError('Tilt limit is 70 degrees')
        if any(abs(t)>70. for t in self.opt5_refine_roll_deg):raise ValueError('OPT5 roll limit is 70 degrees')
        if any(d<.005 or d>.10 for d in self.opt5_approach_lateral_m):raise ValueError('OPT5 approach lateral bounds')
        if any(abs(d)>.005 for d in self.adaptive_mount_x_offsets_m):raise ValueError('Adaptive mount shift must be within 5mm')
        if any(abs(d)>10. for d in self.opt7_jaw_shift_mm):raise ValueError('OPT7 jaw shift limit is 10 mm')
        if self.halo_objects is not None and len(set(self.halo_objects))!=len(self.halo_objects):
            raise ValueError('Duplicate halo_objects')
        return self

