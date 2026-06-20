from __future__ import annotations

import csv
import json
import math
import re
import statistics
import struct
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _find_repo_root(start: Path) -> Path:
    """Cari root repo yang berisi folder `data/` (atau `.git`), naik dari lokasi file ini.

    Robust terhadap lokasi script: baik saat berada di `configs/scripts/` maupun setelah
    dipindah ke `src/`. Fallback ke parents[1] jika tak ketemu.
    """
    for parent in [start, *start.parents]:
        if (parent / "data").is_dir() or (parent / ".git").is_dir():
            return parent
    return start.parents[1] if len(start.parents) >= 2 else start


ROOT = _find_repo_root(Path(__file__).resolve())
DATA_DIR = ROOT / "data"
CONFIG_DIR = ROOT / "configs"
FEATURE_DIR = ROOT / "features"
REPORT_DIR = ROOT / "reports"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


COBB500_AS_HATCHED: Dict[int, int] = {
    0: 42,
    1: 55,
    2: 71,
    3: 90,
    4: 112,
    5: 138,
    6: 168,
    7: 202,
    8: 240,
    9: 283,
    10: 330,
    11: 382,
    12: 440,
    13: 503,
    14: 570,
    15: 639,
    16: 711,
    17: 786,
    18: 864,
    19: 945,
    20: 1029,
    21: 1116,
    22: 1205,
    23: 1296,
    24: 1390,
    25: 1486,
    26: 1583,
    27: 1682,
    28: 1783,
    29: 1886,
    30: 1989,
    31: 2094,
    32: 2200,
    33: 2306,
    34: 2413,
    35: 2521,
    36: 2629,
    37: 2738,
    38: 2846,
    39: 2954,
    40: 3062,
    41: 3170,
    42: 3278,
    43: 3384,
    44: 3490,
    45: 3595,
    46: 3699,
    47: 3801,
    48: 3902,
    49: 4001,
    50: 4099,
    51: 4195,
    52: 4289,
    53: 4380,
    54: 4470,
    55: 4557,
    56: 4641,
}


def ensure_dirs() -> None:
    for path in [CONFIG_DIR, FEATURE_DIR, REPORT_DIR, REPORT_DIR / "plots", REPORT_DIR / "overlays"]:
        path.mkdir(parents=True, exist_ok=True)


def read_xlsx_first_sheet(path: Path) -> List[Dict[str, str]]:
    """Small XLSX reader for simple shared-string sheets. Avoids pandas/openpyxl."""
    ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path) as z:
        shared: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join((t.text or "") for t in si.findall(".//a:t", ns)))

        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
        parsed_rows: List[Dict[int, str]] = []
        for row in sheet.findall(".//a:sheetData/a:row", ns):
            values: Dict[int, str] = {}
            for cell in row.findall("a:c", ns):
                ref = cell.attrib.get("r", "A1")
                col_letters = re.sub(r"\d+", "", ref)
                col_idx = 0
                for ch in col_letters:
                    col_idx = col_idx * 26 + (ord(ch.upper()) - ord("A") + 1)
                col_idx -= 1

                cell_type = cell.attrib.get("t")
                value = ""
                if cell_type == "inlineStr":
                    value = "".join((t.text or "") for t in cell.findall(".//a:t", ns))
                else:
                    node = cell.find("a:v", ns)
                    if node is not None and node.text is not None:
                        value = node.text
                        if cell_type == "s":
                            value = shared[int(value)]
                values[col_idx] = value
            if values:
                parsed_rows.append(values)

    if not parsed_rows:
        return []
    max_col = max(max(r.keys()) for r in parsed_rows)
    headers = [parsed_rows[0].get(i, "").strip() for i in range(max_col + 1)]
    rows: List[Dict[str, str]] = []
    for row in parsed_rows[1:]:
        item = {headers[i]: row.get(i, "").strip() for i in range(len(headers)) if headers[i]}
        if any(item.values()):
            rows.append(item)
    return rows


def parse_filename_metadata(name: str) -> Dict[str, Any]:
    m = re.match(r"^(?P<code>[CP])-W(?P<week>\d+)-?(?P<rest>.*)$", Path(name).stem, re.IGNORECASE)
    if not m:
        return {"house_code": None, "house": None, "week": None, "age_days": None}
    code = m.group("code").upper()
    week = int(m.group("week"))
    return {
        "house_code": code,
        "house": "Commercial" if code == "C" else "Prototype" if code == "P" else None,
        "week": week,
        "age_days": week * 7,
    }


def cobb_weight_for_age(age_days: int) -> Optional[int]:
    return COBB500_AS_HATCHED.get(age_days)


def iter_images(split: str) -> List[Path]:
    p = DATA_DIR / "images" / split
    if not p.exists():
        return []
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS)


def iter_labels(split: str) -> List[Path]:
    p = DATA_DIR / "labels" / split
    if not p.exists():
        return []
    return sorted(x for x in p.iterdir() if x.is_file() and x.suffix.lower() == ".txt")


def read_yolo_label(path: Path) -> List[Tuple[int, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO columns, got {len(parts)}")
        cls = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        rows.append((cls, x, y, w, h))
    return rows


def image_size(path: Path) -> Tuple[int, int]:
    suffix = path.suffix.lower()
    with path.open("rb") as f:
        data = f.read(32)
        if suffix == ".png" or data.startswith(b"\x89PNG\r\n\x1a\n"):
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError(f"Not a PNG: {path}")
            return struct.unpack(">II", data[16:24])
        if data[:2] == b"\xff\xd8":
            return jpeg_size(path)
    raise ValueError(f"Unsupported image format for size read: {path}")


def jpeg_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as f:
        if f.read(2) != b"\xff\xd8":
            raise ValueError(f"Not a JPEG: {path}")
        while True:
            marker_prefix = f.read(1)
            if not marker_prefix:
                break
            if marker_prefix != b"\xff":
                continue
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            if marker in [b"\xd8", b"\xd9"]:
                continue
            length_bytes = f.read(2)
            if len(length_bytes) != 2:
                break
            length = struct.unpack(">H", length_bytes)[0]
            if marker[0] in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                seg = f.read(5)
                if len(seg) != 5:
                    break
                height, width = struct.unpack(">HH", seg[1:5])
                return width, height
            f.seek(length - 2, 1)
    raise ValueError(f"Cannot read JPEG dimensions: {path}")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def median(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.median(vals) if vals else None


def mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.mean(vals) if vals else None


def stdev(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    return statistics.stdev(vals) if len(vals) >= 2 else 0.0


def percentile(values: Iterable[float], p: float) -> Optional[float]:
    vals = sorted(v for v in values if math.isfinite(v))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    k = (len(vals) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return vals[int(k)]
    return vals[lo] * (hi - k) + vals[hi] * (k - lo)


def group_by(rows: Iterable[Dict[str, Any]], key: str) -> Dict[Any, List[Dict[str, Any]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row)
    return dict(grouped)


def float_fmt(v: Any, digits: int = 4) -> Any:
    if isinstance(v, float):
        return round(v, digits)
    return v