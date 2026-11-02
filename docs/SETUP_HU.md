# Hozzáadás GitHubhoz – magyar lépések

Ez a csomag **nem** tartalmazza a teljes Python-forrást, gyártói CAD fájlokat és RGB-D capture-öket. A már meglévő projekt mellé kell beilleszteni.

1. Klónozd a már létrehozott `GPagnostic-dev` repót a saját gépedre GitHub Desktop segítségével vagy `git clone` paranccsal.
2. A starter ZIP tartalmát másold a klónozott repo gyökerébe. Ellenőrizd a README-t és a `.gitignore`-t.
3. A tényleges, működő Python-projektből másold át a `grasp_planner/`, `configs/`, `scripts/`, `tests/`, `pyproject.toml` és `NOTICE.md` fájlokat/mappákat.
4. Ne tölts fel `.venv/`-et, személyes meghajtóútvonalakat, teljes RGB-D datasetet vagy több száz MB generált riportot.
5. Vedd fel az eredeti változatok pontos konfigurációját, commitját és mérési eredményének hashét az `experiments/` regiszterbe.
6. Először lokálisan: `git status`, `git diff --stat`, `git add README.md .gitignore .gitattributes docs experiments data/README.md data/manifests results/summaries`, majd a forrásfájlok hozzáadása után commit/push.
7. Ne taggeld az OPT4-et sikeres verziónak addig, amíg a teljes kísérlet és összehasonlítás nem készült el.

Példa **teljes PowerShell futtatásra** a meglévő capture `0000`-val, ha a Python-forrás már a klónozott repo gyökerében van:

```powershell
$repo = 'C:\Users\v1nce\Desktop\ProjektGP\GPagnostic-dev'
$dataset = 'C:\Users\v1nce\Desktop\ProjektGP\Hand_tools\0000'
$python = Join-Path $repo '.venv\Scripts\python.exe'
Set-Location -LiteralPath $repo
if (-not (Test-Path -LiteralPath $python)) {
    python -m venv '.venv'
    if ($LASTEXITCODE -ne 0) { throw 'Venv creation failed.' }
}
& $python -m pip install -e '.[test]'
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $python -m pytest -q
if ($LASTEXITCODE -ne 0) { throw 'Tests failed.' }
& $python -m grasp_planner.cli run `
  --dataset $dataset `
  --config (Join-Path $repo 'configs\planner.yaml') `
  --gripper (Join-Path $repo 'configs\grippers\2fg7.yaml') `
  --gripper (Join-Path $repo 'configs\grippers\rg2.yaml') `
  --gripper (Join-Path $repo 'configs\grippers\rg6.yaml') `
  --output (Join-Path $repo 'local_results\capture_0000_run')
if ($LASTEXITCODE -ne 0) { throw 'Planner run failed.' }
```

A parancs az aktuális `configs/planner.yaml` szerinti modellt futtatja (amely OPT4 esetén kísérleti is lehet), **nem** a konzervatív baseline-t. A fájlból ellenőrizd a `halo_policy` és `target_contact_policy` mezőket.
