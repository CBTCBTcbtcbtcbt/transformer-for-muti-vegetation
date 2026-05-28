from __future__ import annotations

import csv
import math
import re
from pathlib import Path

ANG_PATTERN = re.compile(r"_ang(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def extract_angle_from_filename(csv_path: str | Path) -> float:
    """Extract clockwise angle (deg) from names like sensor_1_ang180.csv."""
    name = Path(csv_path).name
    match = ANG_PATTERN.search(name)
    if not match:
        raise ValueError(f"Cannot parse angle from filename: {name}")
    return float(match.group(1))


def rotate_force_xy_to_zero_frame(fx: float, fy: float, angle_deg: float) -> tuple[float, float]:
    """
    Rotate measured Fx/Fy back to the 0-degree sensor frame.

    Convention:
    - angle_deg is clockwise-positive (top view).
    - Fy positive points west.
    """
    rad = math.radians(angle_deg)
    c = math.cos(rad)
    s = math.sin(rad)

    fx_0 = fx * c + fy * s
    fy_0 = -fx * s + fy * c
    return fx_0, fy_0


def rotate_force_csv_to_zero_frame(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    angle_deg: float | None = None,
    *,
    delimiter: str = ",",
    fx_col: int = -3,
    fy_col: int = -2,
    has_header: bool = False,
) -> Path:
    """
    Read one CSV and rotate its Fx/Fy columns into the 0-degree frame.

    By default, Fx/Fy are assumed to be the last 3rd and last 2nd columns.
    """
    input_path = Path(input_csv)
    if angle_deg is None:
        angle_deg = extract_angle_from_filename(input_path)

    rows: list[list[str]] = []
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        rows = [row for row in reader if row]

    if not rows:
        raise ValueError(f"CSV is empty: {input_path}")

    start_row = 1 if has_header else 0
    sample = rows[start_row]
    if len(sample) < 3:
        raise ValueError(f"CSV must have at least 3 columns, got {len(sample)}: {input_path}")

    for i in range(start_row, len(rows)):
        row = rows[i]
        fx = float(row[fx_col])
        fy = float(row[fy_col])
        fx_0, fy_0 = rotate_force_xy_to_zero_frame(fx, fy, angle_deg)
        row[fx_col] = f"{fx_0:.10f}"
        row[fy_col] = f"{fy_0:.10f}"

    if output_csv is None:
        output_csv = input_path.with_name(f"{input_path.stem}_rot0{input_path.suffix}")
    output_path = Path(output_csv)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerows(rows)
    return output_path


def batch_rotate_sensor_csvs_to_zero_frame(
    data_root: str | Path = Path("data"),
    output_root: str | Path = Path("data") / "datanew",
) -> list[Path]:
    """
    Batch-rotate all raw sensor angle CSV files and write to data/datanew.

    Keeps relative folder structure under data_root, e.g.:
    data/yp1/sensor_1_ang60.csv -> data/datanew/yp1/sensor_1_ang60.csv
    """
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    processed: list[Path] = []
    for src in sorted(data_root.rglob("sensor_*_ang*.csv")):
        if output_root in src.parents:
            continue
        if src.name.endswith("_filtered.csv") or src.name.endswith("_rot0.csv"):
            continue

        rel = src.relative_to(data_root)
        dst = output_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        rotate_force_csv_to_zero_frame(src, dst)
        processed.append(dst)

    return processed


if __name__ == "__main__":
    outputs = batch_rotate_sensor_csvs_to_zero_frame()
    print(f"Processed files: {len(outputs)}")
    print("Output root: data/datanew")
