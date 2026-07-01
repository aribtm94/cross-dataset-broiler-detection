from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def run(script: str) -> None:
    print(f"\n=== {script} ===")
    subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=ROOT)


def main() -> None:
    # Fitur & estimasi berat (pipeline lama)
    run("audit_dataset.py")
    run("extract_bbox_features.py")
    run("estimate_weight_anomalies.py")
    run("compare_camera_corrections.py")
    run("xue_light_calibration.py")
    run("image_level_anomaly.py")
    # Anomali metode baru (ensemble) + perbandingan
    run("anomaly_ensemble.py")
    run("anomaly_compare.py")
    print("\nDone. Open reports/anomaly_report.html & reports/anomaly_method_comparison.html")


if __name__ == "__main__":
    main()