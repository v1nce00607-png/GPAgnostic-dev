# Frontend API

FastAPI alkalmazás: `grasp_planner.api:app`. A `docs/openapi.json`, `GeometryConfig.schema.json`, `HardwareSpec.schema.json` használható űrlapgeneráláshoz. Szigorú konfiguráció: ismeretlen mezők és nem véges számok hibát adnak.

```json
{"dataset_id":"0000","planner":{"reference_frame":0,"bbox_scale":1.25,"occupancy_min_views":3},"grippers":[{"model":"2fg7","name":"OnRobot_2FG7","contact_inset_m":0.002},{"model":"rg2","name":"OnRobot_RG2"},{"model":"rg6","name":"OnRobot_RG6"}]}
```

A `POST /v3/plan` 202 választ ad `job_id` és `status_url` mezővel. Kérdezd le a státuszt: queued/running/completed/failed. A completed a számítás végét jelenti, nem sikeres graspot. A JSON állapot atomikusan íródik.

A kész futásban: summary.json, top10_grasps.json, all_candidates.json, report.html, scene_explorer.html, figures/, trajectories/, models_3d/. A HTML-ek relatív fájlhivatkozásait őrizd meg.

A felületen együtt jelenjen meg a `status`, `validation_scope`, `checks` és `executable`. Az environment PASS nem alakítható teljes PASS-ra. FAIL pályát kizárólag diagnosztikaként szabad megjeleníteni. A szoftver nem küld robotparancsot.

A kamera OpenCV rendszerű: X jobbra, Y lefelé, Z előre. Méter, 4×4 homogén transzformációk, XYZW kvaterniók. Az RG kontaktközéppontja nyitáskor változik; a zárás talppóza fix. A megközelítés a tényleges nyitott kontaktközéppontot exportálja.

A három beépített hardverprofil paraméterezhető. Önkényes URDF-feltöltés és tetszőleges többujjas kéz nem támogatott ebben a kiadásban. A szolgáltatás lokális kutatási használatra készült; nyilvános telepítéshez hitelesítés és erőforrás-korlátozás szükséges.
