# Data storage

**No raw dataset is bundled with this repository starter.** Large captures and generated results are excluded by `.gitignore`.

Recommended local-only layout:

```text
data/
└── 0000/
    ├── camera.json
    ├── images/
    ├── depth_maps/
    ├── sam6d_scene/
    └── reconstruction/
```

The actual capture reader may require additional CAD and pose/annotation files. Preserve the original source layout when making an authorized local copy. Publish only metadata under `data/manifests/` after validating access and redistribution rights.
