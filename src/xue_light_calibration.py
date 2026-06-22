from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

from common import CONFIG_DIR, REPORT_DIR, ensure_dirs, iter_images, write_json


def main() -> None:
    ensure_dirs()
    report: Dict[str, Any] = {
        "method": "Xue-light straight-line calibration probe",
        "paper_basis": "Learning to Calibrate Straight Lines for Fisheye Image Rectification, CVPR 2019",
        "status": "not_run",
        "note": "Optional diagnostic. Uses long straight-line support as a proxy for whether Xue/plumb-line calibration is feasible in this barn dataset.",
    }
    try:
        import cv2  # type: ignore
    except Exception as exc:
        report.update(
            {
                "status": "opencv_unavailable",
                "error": str(exc),
                "recommendation": "Install opencv-python to run line detection. Pipeline can continue without Xue-light calibration.",
            }
        )
        write_json(REPORT_DIR / "xue_light_calibration.json", report)
        write_json(CONFIG_DIR / "xue_light_calibration.json", report)
        print("Wrote reports/xue_light_calibration.json (opencv unavailable)")
        return

    images = (iter_images("train")[:12] + iter_images("val")[:8])[:20]
    samples: List[Dict[str, Any]] = []
    total_lines = 0
    total_long_lines = 0
    weighted_angle = 0.0
    weighted_length = 0.0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 60, 160)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180, threshold=120, minLineLength=180, maxLineGap=25)
        line_count = 0 if lines is None else len(lines)
        long_lines = []
        if lines is not None:
            for line in lines[:, 0, :]:
                x1, y1, x2, y2 = map(float, line)
                length = math.hypot(x2 - x1, y2 - y1)
                if length >= 250:
                    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                    long_lines.append((length, angle))
                    weighted_angle += abs(angle) * length
                    weighted_length += length
        total_lines += line_count
        total_long_lines += len(long_lines)
        samples.append(
            {
                "image": img_path.name,
                "line_segments": line_count,
                "long_line_segments": len(long_lines),
                "mean_abs_angle_deg": round(sum(abs(a) for _, a in long_lines) / len(long_lines), 2) if long_lines else None,
            }
        )

    support_score = total_long_lines / max(1, len(samples))
    report.update(
        {
            "status": "completed",
            "sampled_images": len(samples),
            "total_line_segments": total_lines,
            "total_long_line_segments": total_long_lines,
            "long_line_support_per_image": round(support_score, 2),
            "weighted_mean_abs_angle_deg": round(weighted_angle / weighted_length, 2) if weighted_length else None,
            "samples": samples,
            "recommendation": "Feasible for Xue/plumb-line calibration" if support_score >= 3 else "Weak line support; prefer radial_depth + image-level robust anomaly",
        }
    )
    write_json(REPORT_DIR / "xue_light_calibration.json", report)
    write_json(CONFIG_DIR / "xue_light_calibration.json", report)
    print("Wrote reports/xue_light_calibration.json")


if __name__ == "__main__":
    main()