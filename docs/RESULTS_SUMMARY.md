# Geometriai kísérleti eredmények

- Egy RGB-D felvétel (318 képkocka), négy tárgy, három megfogó, 600 jelölt kombinációnként.
- 7200 jelöltből 935 teljes geometriai PASS és 6265 FAIL.
- 176 különböző póz; a külön ellenőrzött kétoldali kontaktusú PASS-jelöltek száma 688, a névleges alapúaké 247.
- PASS megfogónként: RG2 448, RG6 229, 2FG7 258.
- Első sikertelen kapu: `target_body` 4486; `environment` 1641; `approach` 78; `closure` 60.
- Az egyedi, külön ellenőrzött kontaktusú pózok száma eltérő összegezésekben 143 vagy 145; ezt publikálás előtt pontosítani kell.
- A mozgáspályák csak a megfogó geometriájára vonatkoznak, nem teljes robotkarra.
- Kísérleti pontfelhő-tisztítás, engedményes voxel/ujjkontaktus feltételek; a PASS feltételes geometriai eredmény.
- Nincs megvizsgált fizikai megfogás, és nincs VLA-finomhangolási eredmény.
