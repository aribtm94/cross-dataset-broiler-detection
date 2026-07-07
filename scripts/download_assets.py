#!/usr/bin/env python3
"""Download, verify, and extract the large assets listed in ``assets_manifest.json``.

The heavy data/weights/vendor files for this project live on Google Drive
(they are too big for Git). This script pulls them, checks the SHA-256 recorded
in the manifest, and unpacks each zip to its declared ``extract_to`` path.

Usage (from repo root)::

    # get everything marked required (datasets, YOLO weights, MOWA+SAM)
    .venv-mowa/Scripts/python.exe scripts/download_assets.py --required

    # get specific bundles
    .venv-mowa/Scripts/python.exe scripts/download_assets.py --only datasets_core weights_yolo

    # get absolutely everything (incl. derived data, features, papers)
    .venv-mowa/Scripts/python.exe scripts/download_assets.py --all

    # list bundles and their status without downloading
    .venv-mowa/Scripts/python.exe scripts/download_assets.py --list

Requires ``gdown`` (already installed in .venv-mowa). Downloaded zips are cached
in ``dist/`` and reused if their hash already matches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "assets_manifest.json"
DIST = ROOT / "dist"

_PLACEHOLDER = "PASTE_GOOGLE_DRIVE_LINK_HERE"


def _load_manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _human(n: int | None) -> str:
    if not n:
        return "?"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _gdown_download(url: str, dest: Path) -> None:
    try:
        import gdown
    except ImportError:
        sys.exit(
            "gdown is not installed. Activate .venv-mowa or run:\n"
            "    pip install gdown"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    # fuzzy=True lets gdown accept full share URLs, not just bare file ids.
    gdown.download(url=url, output=str(dest), quiet=False, fuzzy=True)


def _verify(path: Path, bundle: dict) -> bool:
    expected = bundle.get("sha256")
    if not expected:
        print(f"  ! no sha256 recorded for '{bundle['id']}' - skipping integrity check")
        return True
    print(f"  verifying sha256 ...")
    actual = _sha256(path)
    if actual != expected:
        print(f"  ! HASH MISMATCH for {path.name}\n      expected {expected}\n      got      {actual}")
        return False
    print("  hash OK")
    return True


def _is_within(child: Path, parent: Path) -> bool:
    """True if resolved ``child`` is ``parent`` or lives inside it.

    Uses path-segment containment (not string prefix) so a sibling dir that
    merely shares the name prefix - e.g. ``repo-EVIL`` vs ``repo`` - is rejected.
    """
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _extract(path: Path, bundle: dict) -> None:
    target = (ROOT / bundle.get("extract_to", ".")).resolve()
    if not _is_within(target, ROOT):
        sys.exit(f"refusing to extract outside repo root: {target}")
    print(f"  extracting -> {target.relative_to(ROOT) or '.'}")
    with zipfile.ZipFile(path) as zf:
        # Guard against path traversal in the archive.
        for name in zf.namelist():
            dest = (target / name).resolve()
            if not _is_within(dest, target):
                sys.exit(f"unsafe path in zip: {name}")
        zf.extractall(target)


def _status(bundle: dict) -> str:
    url = bundle.get("gdrive_url", "")
    if not url or url == _PLACEHOLDER:
        return "no-link"
    cached = DIST / bundle["file"]
    if cached.exists():
        return "cached"
    return "ready"


def select(bundles: list[dict], args: argparse.Namespace) -> list[dict]:
    if args.only:
        wanted = set(args.only)
        chosen = [b for b in bundles if b["id"] in wanted]
        unknown = wanted - {b["id"] for b in bundles}
        if unknown:
            sys.exit(f"unknown bundle id(s): {', '.join(sorted(unknown))}")
        return chosen
    if args.all:
        return bundles
    if args.required:
        return [b for b in bundles if b.get("required")]
    return []


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--required", action="store_true", help="download required bundles (default set)")
    g.add_argument("--all", action="store_true", help="download every bundle")
    g.add_argument("--only", nargs="*", help="download specific bundle ids")
    ap.add_argument("--list", action="store_true", help="list bundles and exit")
    ap.add_argument("--no-extract", action="store_true", help="download+verify but do not unpack")
    ap.add_argument("--force", action="store_true", help="re-download even if a valid cached zip exists")
    args = ap.parse_args(argv)

    data = _load_manifest()
    bundles = data["bundles"]

    if args.list or not (args.required or args.all or args.only):
        print(f"{'id':<24} {'req':<4} {'size':>9}  {'status':<8} title")
        print("-" * 78)
        for b in bundles:
            req = "yes" if b.get("required") else "-"
            print(f"{b['id']:<24} {req:<4} {_human(b.get('size_bytes')):>9}  {_status(b):<8} {b['title']}")
        if not args.list:
            print("\nNothing selected. Use --required, --all, or --only <id ...>.")
        return 0

    chosen = select(bundles, args)
    DIST.mkdir(exist_ok=True)
    failures = []

    for b in chosen:
        print(f"\n=== {b['id']}: {b['title']} ({_human(b.get('size_bytes'))}) ===")
        url = b.get("gdrive_url", "")
        if not url or url == _PLACEHOLDER:
            print("  ! no gdrive_url set in manifest yet - skipping")
            failures.append(b["id"])
            continue

        dest = DIST / b["file"]
        if dest.exists() and not args.force and _verify(dest, b):
            print("  using cached download")
        else:
            _gdown_download(url, dest)
            if not _verify(dest, b):
                failures.append(b["id"])
                continue

        if not args.no_extract:
            _extract(dest, b)

    print("\nDone.", f"Failed: {', '.join(failures)}" if failures else "All selected bundles OK.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
