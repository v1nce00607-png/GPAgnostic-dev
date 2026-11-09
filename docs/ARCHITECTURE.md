# Forráskód-architektúra

## Adatfolyam

RGB-D → rekonstrukció → jelenettisztítás → CAD/URDF → jelöltgenerálás → kontaktusoptimalizálás → tíz ellenőrzés → mozgáspálya → riport.

## A fő modulok szerepe

| Modul | Felelősség | Fő belépési függvények |
|---|---|---|
| `__init__.py` | Python-csomag inicializálása |  |
| `adaptive_contact.py` | Egytengelyes kontaktuskezdőpont és adattávolság keresése | `local_contact_obb()`, `fingertip_anchor_distances()`, `solve_adaptive_closure()` |
| `api.py` | REST API, feladatindítás és eredmény-hozzáférés | `PlanRequest`, `roots()`, `jobpath()`, `health()`, `schemas()`, `validate()` |
| `approach_rrt.py` | RRT-Connect faépítés, él- és rövidítésellenőrzés | `_Tree`, `rrt_connect()` |
| `cad_repair.py` | Nyitott CAD-hálók javítása és konvex felbontás | `repair_to_solid()`, `audit_repair()`, `decompose()`, `audit_decomposition()`, `target_solids()`, `gripper_link_solid()` |
| `cleaning.py` | SOR, DBSCAN, sugaras szűrés, halo-tisztítás és síkok | `preprocess()`, `relabel_dominance()`, `floating_geometry_candidates()`, `floating_evidence_probe()`, `floating_depth_confirmed()`, `halo_roi_candidates()` |
| `cli.py` | Parancssori belépési pont | `main()` |
| `contact_goal_opt.py` | Korlátos kétoldali OEM ujjkontaktus-illesztés | `original_pad_anchor_residuals()`, `_Budget`, `optimize_contact_goal()` |
| `fcl_persistent.py` | FCL-objektumok tartós gyorsítótárazása | `PosedMesh`, `closest_distance()`, `PersistentCollisionManager` |
| `geometry_collision.py` | Statikus, környezeti és söpört ütközési feltételek | `swept()`, `local_contact_witnesses()`, `witness_summary()`, `Scene` |
| `geometry_config.py` | Paraméterek és érvényes tartományok sémája | `GeometryConfig` |
| `geometry_planner.py` | CAD-sugár jelöltek, kiválasztás, kapuk, megközelítés | `generate()`, `candidate_target_check()`, `nonpad_target_surface_clearance()`, `select()`, `rank_key()`, `approach()` |
| `geometry_report.py` | HTML/PNG/SVG/3D megjelenítések | `render()`, `supplement()` |
| `models.py` | Képek, maszkok, kameraállások és tárgymodellek beolvasása | `native()`, `save()`, `digest()`, `transform()`, `pose()`, `valid_transform()` |
| `opt7_diagnostics.py` | Ütközési tanúpontok és hibadiagnosztika | `flip180_mount()`, `colliding_links()`, `opt7_candidate_diagnostics()` |
| `pipeline.py` | Rekonstrukció, tisztítás, futtatás és export vezérlése | `load_reconstruction()`, `runtime_hardware()`, `store_diagnostic_checks()`, `finish()`, `run()` |
| `reconstruction.py` | Mélység-visszavetítés és többnézetes fúzió | `Reconstruction`, `backproject()`, `reduce()`, `fit_plane()`, `reconstruct()`, `alignment()` |
| `static_pair.py` | Nyitott pregrasp-póz keresése, nem igazolt kontaktus | `find_joint_static_pregrasp()` |
| `trajectory.py` | Idő szerinti útvonalmintavételezés | `timed_path()` |
| `ur_hardware.py` | Gyártói URDF/mesh kinematika és ujjfelületek | `HardwareSpec`, `origin()`, `Part`, `resources()`, `CADGripper` |

## Mérés és naplózás

A tíz kapu sorrendje: `data_alignment`, `hardware_geometry`, `kinematics`, `contact_geometry`, `self_collision`, `target_body`, `environment`, `approach`, `closure`, `filter_policy`.

A geometriai PASS nem egyenértékű az igazolt fizikai megfogással. A kontakteredetet külön érdemes szűrni (`contact_basis`).

## Segédprogramok áttekintése

| Script | Vizsgálat |
|---|---|
| `compare_runs.py` | Compare two complete grasp experiments (aggregate metrics, not candidate IDs). |
| `evidence_ablation.py` | Read-only audit of recorded occupancy witnesses against multiview evidence. |
| `expanded_search.py` | Repeat wider contact search and extreme bbox deletion ablation. Partial checks only. |
| `geometry_controls.py` | Known free/blocked geometric controls using all three OEM CAD grippers. |
| `opt2_local_refinement.py` | Revalidate local grasp-pose offsets against the original geometric gates. |
| `opt3_floating_visualize.py` | Create an HTML 3D plot of retained, removed and uncertainty-evidence voxels. |
| `opt3_metrics.py` | Generate transparent OPT1-vs-OPT3 metrics and contact-model provenance. |
| `opt4_metrics.py` | Report OPT3/OPT4 PASS values with prominent experimental-scene caveat. |
| `opt4_preview.py` | Quick, read-only preview of OPT4 object-space ROI on saved OPT3 points. |
| `opt4_roi_visualize.py` | Original 3D samples colored by suppression and target label; independent of camera up/down. |
| `opt4_run_cached.py` | Run full OPT4 validation, reusing IDENTICAL OPT3 raw RGB-D voxel fusion. |
| `opt5_adaptive_metrics.py` | Audit the OPT5 CAD-guided experiment without confusing conditional geometry with certification. |
| `opt5_cad_audit.py` | Audit OnRobot 2FG7 manufacturer URDF visual/collision meshes in-place. |
| `opt5_metrics.py` | Compare OPT4 and OPT5 scientific gate statuses; exclude diagnostic-only paths. |
| `opt5_run_cached.py` | Run full OPT5 validation, reusing IDENTICAL OPT4 raw RGB-D voxel fusion. |
| `opt7_audit.py` | OPT7 post-hoc audit of any all_candidates.json (no re-planning, read-only). |
| `opt7_combined_report.py` | Merge per-gripper runs (same dataset/config) into ONE run folder and render the |
| `opt7_compare_report.py` | Comparison report (Hungarian, single HTML): the version before vs the version built with the latest changes. |
| `opt7_report.py` | OPT7 best-case report (single self-contained HTML, Hungarian). |
| `opt7_tdk_report.py` | TDK report (Hungarian, single HTML): the final configuration (Q2) and how the framework got there. |
| `opt7_verify_equivalence.py` | Re-evaluate the STORED final pose of every candidate with the current collision backend and compare |
| `prepare_data.py` | Verify and extract user-provided archives, including split RGB source archive. |
| `resume_opt1_report.py` | Diagnosztikai vagy kísérleti eszköz; az argumentumok a scriptben találhatók. |
| `verify_release.py` | Verify OPT1 source manifest and model-relative candidate status integrity. |

## Tesztek

A projektben 19 tesztmodul szerepel. A tesztek igazolják a kódon belüli szerződés egy részét, de nem robotkísérletek. Az aktuális csomag létrehozásakor a teszteket nem futtattam újra.
