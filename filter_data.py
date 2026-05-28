from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt



def filter_data_array(data: np.ndarray, fs: float, fc: float, trim_head: int, trim_tail: int) -> np.ndarray:
    """Trim head/tail rows and apply 4th-order Butterworth low-pass filtering column-wise."""
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError("data must be numeric")

    if fs <= 0:
        raise ValueError("fs must be > 0")
    if not (0 < fc < fs / 2):
        raise ValueError("fc must satisfy 0 < fc < fs/2")
    if trim_head < 0 or trim_tail < 0:
        raise ValueError("trim_head and trim_tail must be >= 0")

    n_rows = data.shape[0]
    if trim_head + trim_tail >= n_rows:
        raise ValueError(
            f"Invalid trimming: trim_head + trim_tail ({trim_head + trim_tail}) must be < number of rows ({n_rows})"
        )

    end_idx = n_rows - trim_tail if trim_tail > 0 else n_rows
    trimmed = data[trim_head:end_idx, :].astype(float, copy=False)

    b, a = butter(4, fc / (fs / 2), btype="low")
    filtered = np.empty_like(trimmed, dtype=float)

    for col in range(trimmed.shape[1]):
        filtered[:, col] = filtfilt(b, a, trimmed[:, col])

    return filtered


def axis_limits(values: np.ndarray, pad_ratio: float = 0.05) -> tuple[float, float]:
    """Calculate y-axis limits with padding."""
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = y_max - y_min
    if span <= 0:
        pad = max(abs(y_max) * pad_ratio, 1e-6)
        return y_min - pad, y_max + pad
    pad = span * pad_ratio
    return y_min - pad, y_max + pad


def main() -> None:
    # Editable hyperparameters
    fs = 1000.0
    fc = 5.0
    trim_head = 600
    trim_tail = 600
    selected_columns = [3]
    show_plot = True

    root = Path(__file__).resolve().parent
    # input_path = root / "data" /"yp1"/ "sensor_1_ang180.csv"
    # output_path = root / "data" /"yp1"/ "sensor_1_ang180_filtered.csv"

    input_path = root / "data" / "datanew" /"yp1"/ "sensor_6_ang300.csv"
    output_path = root / "data" /"datanew" /"yp1"/ "sensor_6_ang300_filtered.csv"
    plot_dir = root / "data"

    data = np.genfromtxt(input_path, delimiter=",")
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    if not selected_columns:
        raise ValueError("selected_columns cannot be empty")
    if min(selected_columns) < 0 or max(selected_columns) >= data.shape[1]:
        raise ValueError(
            f"selected_columns out of range. Valid range is [0, {data.shape[1] - 1}]"
        )

    print("=== Normal test ===")
    print(f"Input file: {input_path}")
    print(f"Input shape: {data.shape}")
    print(
        "Hyperparameters: "
        f"fs={fs}, fc={fc}, trim_head={trim_head}, trim_tail={trim_tail}, "
        f"selected_columns={selected_columns}"
    )

    end_idx = data.shape[0] - trim_tail if trim_tail > 0 else data.shape[0]
    trimmed_full = data[trim_head:end_idx, :].astype(float, copy=False)

    selected_data = data[:, selected_columns]
    filtered_selected = filter_data_array(selected_data, fs, fc, trim_head, trim_tail)

    filtered = trimmed_full.copy()
    filtered[:, selected_columns] = filtered_selected
    np.savetxt(output_path, filtered, delimiter=",", fmt="%.10f")

    expected_rows = data.shape[0] - trim_head - trim_tail
    print(f"Output file: {output_path}")
    print(f"Output shape: {filtered.shape}")
    print(f"Expected rows after trim: {expected_rows}")

    print("\n=== Visualization ===")
    figs = []
    for col in selected_columns:
        before_path = plot_dir / f"sensor_1_col_{col}_before.png"
        after_path = plot_dir / f"sensor_1_col_{col}_after.png"
        before_y_min, before_y_max = axis_limits(trimmed_full[:, col])
        after_y_min, after_y_max = axis_limits(filtered[:, col])
        x_min = 0
        x_max = trimmed_full.shape[0] - 1

        fig_before, ax_before = plt.subplots(figsize=(12, 3.5))
        ax_before.plot(trimmed_full[:, col], color="tab:blue", linewidth=1.0)
        ax_before.set_title(f"Column {col} - Before filter")
        ax_before.set_xlabel("Sample index (after trim)")
        ax_before.set_ylabel("Value")
        ax_before.set_xlim(x_min, x_max)
        ax_before.set_ylim(before_y_min, before_y_max)
        ax_before.grid(True, alpha=0.3)
        fig_before.tight_layout()
        fig_before.savefig(before_path, dpi=150)
        print(f"Saved plot: {before_path}")
        figs.append(fig_before)

        fig_after, ax_after = plt.subplots(figsize=(12, 3.5))
        ax_after.plot(filtered[:, col], color="tab:orange", linewidth=1.0)
        ax_after.set_title(f"Column {col} - After filter")
        ax_after.set_xlabel("Sample index (after trim)")
        ax_after.set_ylabel("Value")
        ax_after.set_xlim(x_min, x_max)
        ax_after.set_ylim(after_y_min, after_y_max)
        ax_after.grid(True, alpha=0.3)
        fig_after.tight_layout()
        fig_after.savefig(after_path, dpi=150)
        print(f"Saved plot: {after_path}")
        figs.append(fig_after)

    if show_plot:
        # Non-blocking display so the script does not wait for window close.
        plt.show(block=False)
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()

