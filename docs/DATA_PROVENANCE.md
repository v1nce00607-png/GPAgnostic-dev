# Data and CAD provenance policy

Do **not** upload the entire `Hand_tools/0000`, `0001`, `0002`, or `0007` captures into ordinary Git history. They include many RGB-D frames, masks, depth maps, camera data and object geometry. The default repository intentionally excludes them.

Record for every dataset and third-party asset:
- Canonical source URL/name, authorized access and redistribution/license status.
- Dataset ID and frame count, calibration revision and annotation revision.
- SHA-256 checksums and capture structure/version.
- Upstream license/NOTICE texts when required.
- Object and gripper CAD/URDF provenance and upstream commit where available.

Current original project NOTICE names the `KarolyArmin/Hand_tools` collection and user-supplied archives; `grasp_planner/assets/source.json` identifies the `OnRobot/onrobot-ros2` repository and a commit. Verify licenses for any assets you actually commit, and preserve required copyright notices.

For approved large files: GitHub blocks ordinary Git objects above 100 MiB, while Git LFS stores pointers in Git and blobs separately. Use `git lfs track 'specific/path/*.npz'` only for selected authorized assets, commit `.gitattributes`, and verify Git LFS billing/quota. Prefer an external versioned dataset repository/archive when distributing the full 318-frame capture.

Avoid committing absolute paths containing usernames, tokens, personal images or organization-private information in public manifests. Use generic capture IDs and reproducible hashes.
