# Módszertan és TDK értelmezés

## Hatókör

Ez a kiadás három gyártói kétujjas profil közös geometriai tervezőjét vizsgálja. A felhasználó kérésére fizikai stabilitás, erőzárás, tömeg/súrlódás és dinamika nincs értékelve. Az eredeti absztrakt ilyen állításai módosítandók: „CAD- és RGB-D-alapú geometriai kontakt-, kinematikai és ütközési vizsgálat, számított megközelítési és zárási pályákkal.” A szintetikus geometriai kontroll nem fizikai szimuláció vagy emelési kísérlet. Tetszőleges többujjas megfogó agnosztikus támogatása további kutatási cél.

## Bemenet és származás

KarolyArmin/Hand_tools, 0000: 318 RGB-D kép, négy példány (Pliers_001, Pliers_002, Rachet_001, Screwdriver_001). A Rachet bemeneti írásmódját megőrizzük. RGB/mélység: 1280×800; uint16 milliméter → méter. A kapott kalibráció és kameratranszformációk használatosak, regisztrációt nem találunk ki.

Az eredeti teljes futásban mind az 1613 forrásfájl SHA-256 ellenőrzése sikerült. A végső összeállítás a mentett, valóban kiszámított 318-frame rekonstrukciót és kész jelöltcsoportokat használja, a hiányzó csoportokat újrafuttatja. A visszaállított 976 szükséges bemeneti fájl hash-e ismét ellenőrizve. A `run_provenance.json` elkülöníti a korábbi teljes és a folytatáskori ellenőrzést. A teljes manifest és rögzített revízió a provenance mappában található. A megjelenítéshez használt referenciakép ugyanebből a hash-sel ellenőrzött revízióból származik.

A CPU, memória/cgroup limit, csomagverziók és threadbeállítások a hardware.json fájlban vannak. Nincs GPU- vagy fizikai robotkísérlet. A gyártói CAD/URDF revízió az assets/source.json fájlban, BSD licenc az assets/LICENSE_OnRobot fájlban szerepel.

## Rekonstrukció és szűrés

Mélységugrás-szűrés: 3×3 szomszédságban 25 mm. Maszkerózió: egy pixel; a levágott perem nem kerül háttérként vissza a fúzióba. Kalibrált visszavetítés, világkoordinátás ROI, képkockánkénti voxelredukció, majd az azonos voxelek megfigyeléseinek átlaga. A támogatás független képkockákat számol. Eltérő nézetek rendezetlen ponttömbjeit nem lehet indexenként átlagolni. A módszer voxel-fúzió, nem TSDF.

SOR kNN átlagtávolság, radius szomszédságszám és DBSCAN kis komponensek külön jelzők. A minimális három képkockás háttér-konszenzus, morfológiai jelzők és többnézetes mélységi ellentmondás alapján kihagyott pontok megmaradnak a filter_evidence.npz és a külön PLY fájlban. Kihagyásuk nem bizonyítja az ismeretlen tér szabadságát. CAD-közeli háttérpont visszacímkézéséhez több nézet mélység- és maszktámogatása kell.

Az 1,25× bbox a lokális keresést szűkíti. A feltétel nélküli bbox-törlés `erase_unverified` módban a filter_policy kaput UNVERIFIED állapotban hagyja, így nem eredményezhet elfogadott teljes PASS-t. A kiegészítő szélsőséges bbox-abláció a régión belüli nem célpontokat hagyja ki, de a többi tárgy CAD-ja, a síkprizmák és az asztal ekkor is megmaradnak.

A domináns asztalsík RANSAC + SVD, a kameraoldal felé orientálva. Az alsó féltere tiltott. További akadálysíkok DBSCAN komponensekre válnak; 2D konvex kontúrjuk vastagított, véges prizma. A támogató voxeladatok is megmaradnak. A síkprizmák hézagot is kitölthetnek, tehát konzervatív elutasítást okozhatnak.

PCA normál: 24 szomszéd, orientáció a medián kamera felé. Elsőrendű MLS csak diagnosztikai export, nem mozgat ütközési mintát. A kontaktkritérium az eredeti CAD sugármetszési normáljait használja.

## Geometria és hardver

KD-fa gyorsítja a voxel-környezet lekérdezését. A félterekkel kiterjesztett konvex burkolat a voxel és a 4 mm szenzorbizonytalanság mellett konzervatív előszűrés. Zárt gyártói linknél pontos CAD-távolság és tartalmazásvizsgálat finomít; nyitott hálónál a konvex burkolat marad. A környezeti clearance 2 mm, a kontakt irányú ujjpár-tolerancia 0,5 mm: konfigurált modellfeltételezések, nem mért teljes szenzorhibamodell.

Mind a négy cél-CAD nem zárt/manifold. A cél és a többi tárgy ezért zárt konvex térfogati burkolatot használ. Ez megakadályozza, hogy a mesh hiányain átjutó hibás geometria PASS-t kapjon, de homorulatba illeszkedő valós megfogást is elutasíthat. A program nem javítja hallgatólagosan a hálókat.

Az URDF teljes joint tree-je, mimic kapcsolatai és gyári limitjei működnek. 2FG7: 19 mm ujjankénti prizmás löket, összesen 38 mm. A stock befelé néző szilikonujj nyílása eltér más szerelési helyzetektől; a 73 mm nem löket. RG2/RG6: forgó kapcsolatok, változó talp–kontaktközép távolság. A zárás talppóza fix, a TCP minden mintában újraszámolt. A CAD-nyílás a 2 mm-es kontaktmélységnél értendő, nem keverhető a katalógus névleges maximummal.

Önütközés: eredeti CAD ellenkező oldali ujjak között, SRDF kivételekkel. A gyári belső mechanizmust a gyári limitek szerint kezeljük; teljes szerelési/mechanikai validálást nem állítunk.

## Tervezés és státuszok

PCA hossztengely minták, magassági kvantilisek, yaw és megközelítési döntés alapján CAD ray-párok készülnek. Valódi nyíláslimit és legfeljebb 35° normáleltérés szűr; axiális round-robin ad térbeli változatosságot. A geometriailag ellenirányú kontakt nem erőzárási bizonyíték.

A 10 cm-es, fix orientációjú megközelítés egyenes és oldalirányú alternatíváit folytonos konvex szakaszburkolat ellenőrzi. A 2FG7 zárási eltolás burkolata a használt linkburkolatokra pontos. Az RG szögszakasz legfeljebb 0,04 rad; a teljes ízületi görbület 1 m/rad² felső korlátjából Δq²/8 bővítés számolódik. Az analitikus burkolat a statikus pontos CAD-nál szigorúbb lehet.

Ötödfokú időskálázás tartja a TCP kinematikai sebesség- és gyorsuláskorlátját. A kimenet a kiválasztott kamera OpenCV rendszerében méter és XYZW kvaternió. Dinamikai erőszámítás nem történik.

Kötelező kapuk: data_alignment, hardware_geometry, kinematics, contact_geometry, self_collision, target_body, environment, approach, closure, filter_policy. Egy FAIL összesített FAIL; hiányzó/ismeretlen kapu mellett nincs teljes PASS. Az unknown volume és a robotkar külön UNVERIFIED, minden terv executable=false. A rangsorolt elutasított jelöltekhez is ténylegesen ellenőrzött, FAIL diagnosztikai pályák exportálhatók.

## Érvényességi korlátok

Egyetlen jelenet négy példánya nem bizonyít általánosítást. Automatikus maszkok/pózok; CAD–mélység konzisztencia nem független ground truth. Nincs fizikai stabilitás, robot–kamera kalibrált UR kar/IK vagy valós robotkísérlet. A nulla teljes PASS nem bizonyítja, hogy a megfogás fizikailag lehetetlen. További munkához auditált maszkok/pózok, zárt tárgygeometria és kiválasztott UR kar kell.

## Elsődleges források

- https://huggingface.co/datasets/KarolyArmin/Hand_tools
- https://github.com/OnRobot/onrobot-ros2
- https://onrobot.com/en/products/2fg7-finger-gripper
- https://onrobot.com/en/products/rg2-finger-gripper
- https://onrobot.com/en/products/rg6-finger-gripper

Forrásellenőrzés: 2026-10-29. A konkrét fájlok SHA-256 értékei a hardver- és adatmanifestekben szerepelnek.
