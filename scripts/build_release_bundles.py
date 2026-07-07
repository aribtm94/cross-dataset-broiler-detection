#!/usr/bin/env python3
"""Build the Google-Drive release bundles declared in ``assets_manifest.json``.

For every bundle in the manifest this script zips its ``sources`` into
``dist/<file>``, then writes the resulting ``size_bytes`` and ``sha256`` back
into the manifest so the numbers stay authoritative. Upload the files in
``dist/`` to Google Drive and paste each share link into the matching
``gdrive_url`` field.

Usage (from repo root)::

    .venv-mowa/Scripts/python.exe scripts/build_release_bundles.py            # build all
    .venv-mowa/Scripts/python.exe scripts/build_release_bundles.py --only datasets_core weights_yolo
    .venv-mowa/Scripts/python.exe scripts/build_release_bundles.py --hash-only # re-hash existing zips

The script only uses the Python standard library, so any interpreter works.
Images and model weights are already compressed, so they are stored (not
re-deflated) to keep the build fast; text-heavy bundles are deflated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets_manifest.json"
DIST = ROOT / "dist"

# Extensions that are already compressed -> store instead of deflate.
_STORED_EXTS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif",
    ".pt", ".pth", ".onnx", ".engine", ".zip", ".pdf",
    ".pptx", ".xlsx", ".7z", ".gz",
}


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _save_manifest(data: dict) -> None:
    MANIFEST.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(source: Path, excludes: Iterable[str]) -> Iterable[Path]:
    """Yield files under ``source`` (a file or dir), skipping excluded segments."""
    exclude_set = set(excludes)

    def is_excluded(p: Path) -> bool:
        return any(part in exclude_set for part in p.relative_to(ROOT).parts)

    if source.is_file():
        if not is_excluded(source):
            yield source
        return
    for p in sorted(source.rglob("*")):
        if p.is_file() and not is_excluded(p):
            yield p


def _zip_bundle(bundle: dict) -> Path:
    out = DIST / bundle["file"]
    DIST.mkdir(exist_ok=True)
    excludes = bundle.get("exclude", [])
    sources = [ROOT / s for s in bundle["sources"]]

    missing = [str(s.relative_to(ROOT)) for s in sources if not s.exists()]
    if missing:
        raise FileNotFoundError(
            f"bundle '{bundle['id']}' is missing sources: {', '.join(missing)}"
        )

    n_files = 0
    total_in = 0
    # allowZip64 for >4 GB bundles.
    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        for source in sources:
            for f in _iter_files(source, excludes):
                arcname = f.relative_to(ROOT).as_posix()
                comp = (
                    zipfile.ZIP_STORED
                    if f.suffix.lower() in _STORED_EXTS
                    else zipfile.ZIP_DEFLATED
                )
                zf.write(f, arcname, compress_type=comp)
                n_files += 1
                total_in += f.stat().st_size
    print(f"  packed {n_files} files ({total_in / 1e9:.2f} GB in) -> {out.name}")
    return out


def build(bundle: dict, hash_only: bool) -> None:
    out = DIST / bundle["file"]
    if bundle.get("prebuilt"):
        # Prebuilt zips already exist at the repo root; copy reference / hash in place.
        src = ROOT / bundle["sources"][0]
        if not src.exists():
            print(f"! skip prebuilt '{bundle['id']}': {src.name} not found")
            return
        out = src  # hash the existing zip where it lives
    elif hash_only:
        if not out.exists():
            print(f"! skip '{bundle['id']}': {out.name} not built yet")
            return
    else:
        print(f"building {bundle['id']} -> dist/{bundle['file']}")
        out = _zip_bundle(bundle)

    size = out.stat().st_size
    print(f"  hashing {out.name} ({size / 1e9:.2f} GB) ...")
    digest = _sha256(out)
    bundle["size_bytes"] = size
    bundle["sha256"] = digest
    print(f"  sha256 = {digest}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", help="build only these bundle ids")
    ap.add_argument("--hash-only", action="store_true", help="re-hash already-built zips, do not repack")
    args = ap.parse_args(argv)

    data = _load_manifest()
    bundles = data["bundles"]
    if args.only:
        wanted = set(args.only)
        bundles = [b for b in bundles if b["id"] in wanted]
        unknown = wanted - {b["id"] for b in bundles}
        if unknown:
            ap.error(f"unknown bundle id(s): {', '.join(sorted(unknown))}")

    for bundle in bundles:
        try:
            build(bundle, args.hash_only)
        except FileNotFoundError as exc:
            print(f"! {exc}")

    _save_manifest(data)
    print(f"\nManifest updated: {MANIFEST.relative_to(ROOT)}")
    print(f"Upload the files in {DIST.relative_to(ROOT)}/ to Google Drive, then paste each")
    print("share link into the matching 'gdrive_url' field in assets_manifest.json.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
