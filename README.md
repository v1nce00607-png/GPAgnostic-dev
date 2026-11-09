# Grasp Planner: szemantikusan annotált 3D jelenetek geometriai megfogástervezése

A kutatási prototípus egy RGB-D alapú rekonstrukcióból, tárgy-CAD modellekből és gyártói megfogómodellekből készít és geometriailag ellenőriz megfogási jelölteket. Három párhuzamos pofás OnRobot megfogóval vizsgáltam: RG2, RG6, 2FG7.

## Tudományos hatókör

A `PASS` azt jelzi, hogy az adott jelölt megfelel a megadott geometriai feltételeknek. Nem igazolja a fizikai megtarthatóságot, a robotkar inverz kinematikáját vagy a valós végrehajtást. Egyes PASS jelöltek nem rendelkeznek önállóan ellenőrzött kétoldali OEM-kontaktussal. A szűrt pontfelhőben alkalmazott tárgyközeli ponttörlés feltételes geometriai döntéseket eredményez.

## Projektfelépítés

```text
grasp_planner/             fő algoritmusok és megfogó-adapterek
  assets/                 OnRobot gyártói modellek és licenc
configs/                  bemeneti és megfogó-konfigurációk
scripts/                  importálás, audit, benchmark, riportok
tests/                    automatizált ellenőrző tesztek
docs/                     architektúra, adatséma, eredmények, API
examples/                 mintakliens
data/                     helyi bemenetek (gitignore)
results/                  helyi eredmények (gitignore)
provenance/               bemeneti eredet és ellenőrzőösszegek
```

## Telepítés

Python 3.10+ és a platformhoz illeszkedő `python-fcl` függőség szükséges.

```powershell
Set-Location 'C:\ProjektGP\GPagnostic-dev'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q tests
```

Az alábbi példa a korábban előkészített adatgyűjteményből fut; a bemeneti mappa elérési útját saját környezetben kell beállítani:

```powershell
Set-Location 'C:\ProjektGP\GPagnostic-dev'
.\.venv\Scripts\python.exe -m grasp_planner.cli run --dataset 'C:\ProjektGP\Hand_tools\0000' --config '.\configs\planner_geometry_final.yaml' --gripper '.\configs\grippers\2fg7.yaml' --gripper '.\configs\grippers\rg2.yaml' --gripper '.\configs\grippers\rg6.yaml' --output 'C:\ProjektGP\GPagnostic_run_new'
```

A konfigurációban szereplő nagyobb feladatok számításigényesek, és az eredeti capture adatai nélkül nem működnek. A tesztek és a tervező nem lettek újrafuttatva ennek a GitHub-csomagnak az előállításakor.

## Dokumentáció

- `docs/ARCHITECTURE.md`: az adatfolyam, modulok és fontos függvények
- `docs/RESULTS_SUMMARY.md`: reprodukálhatóan értelmezhető eredmények
- `docs/METHOD_AND_TDK.md`: eredeti részletes technikai háttér
- `NOTICE.md`: adatok és harmadik féltől származó tartalmak eredete

A repó előkészített helyi projekt, nem került automatikusan GitHubra feltöltésre. A közzététel előtt ellenőrizd a saját kód licencelését és minden mellékelt gyártói adat felhasználási feltételeit.
