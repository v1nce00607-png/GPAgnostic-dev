"""Verify and extract user-provided archives, including split RGB source archive."""

from pathlib import Path
import argparse, json, zipfile, hashlib, shutil, stat


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def safe_extract(archive, destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if sum(i.file_size for i in z.infolist()) > 4_000_000_000:
            raise ValueError("Extraction budget exceeded")
        for i in z.infolist():
            p = (root / i.filename).resolve()
            if not p.is_relative_to(root) or stat.S_ISLNK(i.external_attr >> 16):
                raise ValueError("Unsafe archive member")
            if i.is_dir():
                p.mkdir(parents=True, exist_ok=True)
                continue
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                h = hashlib.sha256()
                with z.open(i) as src:
                    while b := src.read(4 << 20):
                        h.update(b)
                if p.stat().st_size != i.file_size or digest(p) != h.hexdigest():
                    raise ValueError("Conflicting duplicate " + i.filename)
            else:
                with z.open(i) as src, p.open("wb") as dst:
                    shutil.copyfileobj(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--uploads", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    for name in [
        "01_capture0000_depth.zip",
        "02_capture0000_semantics_poses.zip",
        "03_capture0000_reconstruction.zip",
    ]:
        safe_extract(a.uploads / name, a.output)
    m = json.loads((a.uploads / "UPLOAD_MANIFEST.json").read_text())
    archive = a.output / "assembled_sources.zip"
    try:
        with archive.open("wb") as dst:
            for part in m["parts"]:
                f = (a.uploads / part["filename"]).resolve()
                if (
                    not f.is_relative_to(a.uploads.resolve())
                    or f.stat().st_size != part["size"]
                    or digest(f) != part["sha256"]
                ):
                    raise ValueError("Bad source part")
                with f.open("rb") as src:
                    shutil.copyfileobj(src, dst)
        if digest(archive) != m["zip_sha256"]:
            raise ValueError("Combined archive hash mismatch")
        safe_extract(archive, a.output)
    finally:
        archive.unlink(missing_ok=True)
    full = json.loads((a.uploads / "capture0000_manifest.json").read_text())
    bad = []
    for r in full["files"]:
        f = (a.output / r["path"]).resolve()
        if (
            not f.is_relative_to(a.output.resolve())
            or not f.is_file()
            or digest(f) != r["sha256"]
        ):
            bad.append(r["path"])
    result = {
        "verified_files": len(full["files"]),
        "errors": bad,
        "all_verified": not bad,
        "revision": full["revision"],
    }
    (a.output / "integrity.json").write_text(json.dumps(result, indent=2))
    print(result)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
