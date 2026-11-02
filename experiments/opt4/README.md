# V4 — aggressive ROI occupancy ablation (not yet verified)

- Goal: test whether object-relative 3D filtering of background points changes Environment PASS counts.
- Configuration includes `halo_policy` (`off`, `isolated_only`, `experimental_roi`), object whitelist and plane/other-object guards.
- `experimental_roi` may remove unlabeled points **without depth-ray proof that the space is empty**.
- At the time of this registry, no successful full OPT4 comparison run was supplied.

Report V4 results separately as **experimental scene-assumption sensitivity**, not equivalent to the conservative benchmark. Record removal counts by object, reviewer inspection of 3D halo visualization and all gate outcomes before publishing metrics.
