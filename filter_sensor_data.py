"""统一的 sensor 原始数据低通/带通滤波脚本。

支持两种模式：
1. lowpass：4 阶 Butterworth 低通，对应旧 filter_data.py 的主要功能。
2. bandpass：Hamming 窗 FIR 带通，对应旧 sensor_data_filter_frequency.py。

项目原始 sensor CSV 为无表头、逗号分隔的 6 列，依次映射为
TX、TY、TZ、FX、FY、FZ。脚本默认批量处理指定文件夹中的
sensor_编号_ang角度.csv，并保持无表头 6 列输出。
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal


# =============================================================================
# 用户可直接修改的默认配置；命令行参数可以临时覆盖这些值。
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
inputpath = PROJECT_ROOT / "data" / "model_1"

# outputpath 为 None 时，输出到 inputpath/sensor_filtered。
outputpath: Path | None = None

filter_mode = "lowpass"
fs = 1000.0
lowpass_cutoff = 5.0
bandpass_low = 0.05
bandpass_high = 1.0
butterworth_order = 4
fir_numtaps = 401
selected_columns = "all"
trim_head = 0
trim_tail = 0
max_nan_gap = 5

SENSOR_COLUMNS = ("TX", "TY", "TZ", "FX", "FY", "FZ")
SENSOR_FILE_RE = re.compile(
    r"^sensor_(?P<index>\d+)_ang(?P<angle>-?\d+(?:\.\d+)?)\.csv$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FilterConfig:
    """保存滤波参数，所有文件共用同一份经过校验的配置。"""

    input_path: Path
    output_dir: Path
    mode: str
    sample_rate: float
    cutoff: float
    band_low: float
    band_high: float
    order: int
    numtaps: int
    columns: tuple[int, ...]
    trim_head: int
    trim_tail: int
    max_nan_gap: int
    make_plots: bool
    overwrite: bool


def parse_sensor_filename(path: Path) -> tuple[int, float] | None:
    """从文件名读取 sensor 编号和角度；不是原始 sensor 文件时返回 None。"""

    match = SENSOR_FILE_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group("index")), float(match.group("angle"))


def discover_sensor_files(path: Path) -> list[Path]:
    """接受单个 CSV 或文件夹，并返回按编号、角度排序的原始 sensor 文件。"""

    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在：{path}")
    if path.is_file():
        if parse_sensor_filename(path) is None:
            raise ValueError(f"文件名不符合 sensor_编号_ang角度.csv：{path.name}")
        return [path]

    files = [item for item in path.iterdir() if item.is_file() and parse_sensor_filename(item)]
    return sorted(files, key=lambda item: (*parse_sensor_filename(item), item.name.lower()))


def read_sensor_csv(path: Path) -> pd.DataFrame:
    """读取原始无表头 6 列 CSV，并明确拒绝 gather 等其他结构。"""

    frame = pd.read_csv(path, header=None, sep=",")
    if frame.empty:
        raise ValueError("文件为空")
    if frame.shape[1] != len(SENSOR_COLUMNS):
        raise ValueError(
            f"检测到 {frame.shape[1]} 列；sensor 原始数据必须为 6 列，"
            "gather CSV 需要使用 data/gather_data.py"
        )

    # errors='coerce' 将无法解析的文本变为 NaN，随后由统一缺失值规则处理。
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    numeric.columns = SENSOR_COLUMNS
    return numeric


def parse_columns(text: str) -> tuple[int, ...]:
    """解析 all、轴名或从 0 开始的列号，例如 all、FX、0,3,5。"""

    if text.strip().lower() == "all":
        return tuple(range(len(SENSOR_COLUMNS)))

    parsed: list[int] = []
    for token in text.split(","):
        item = token.strip()
        if not item:
            continue
        if item.upper() in SENSOR_COLUMNS:
            index = SENSOR_COLUMNS.index(item.upper())
        else:
            try:
                index = int(item)
            except ValueError as error:
                raise ValueError(f"未知列 {item!r}；请使用 TX..FZ 或 0..5") from error
        if index < 0 or index >= len(SENSOR_COLUMNS):
            raise ValueError(f"列号 {index} 越界；合法范围是 0..5")
        if index not in parsed:
            parsed.append(index)

    if not parsed:
        raise ValueError("至少需要选择一列")
    return tuple(parsed)


def trim_frame(frame: pd.DataFrame, head: int, tail: int) -> pd.DataFrame:
    """裁掉首尾指定行数，并检查结果仍有数据。"""

    if head < 0 or tail < 0:
        raise ValueError("trim_head 和 trim_tail 不能为负数")
    if head + tail >= len(frame):
        raise ValueError(
            f"裁剪行数 {head}+{tail} 必须小于原始行数 {len(frame)}"
        )
    end = len(frame) - tail if tail else len(frame)
    return frame.iloc[head:end].reset_index(drop=True).copy()


def fill_small_gaps(values: np.ndarray, max_gap: int, column_name: str) -> np.ndarray:
    """只插值较短的内部 NaN/Inf 缺口，避免悄悄修复严重损坏的数据。"""

    series = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan)
    missing = series.isna().to_numpy()
    if not missing.any():
        return series.to_numpy()
    if missing[0] or missing[-1]:
        raise ValueError(f"{column_name} 的首尾存在 NaN/Inf，无法安全插值")

    # 找出连续缺失区间长度；任一区间过长就停止处理该文件。
    padded = np.concatenate(([False], missing, [False])).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    longest_gap = int(np.max(ends - starts))
    if longest_gap > max_gap:
        raise ValueError(
            f"{column_name} 最长缺失区间为 {longest_gap}，超过允许值 {max_gap}"
        )
    return series.interpolate(method="linear").to_numpy()


def validate_filter_config(config: FilterConfig) -> None:
    """在读取大量文件前检查所有与文件长度无关的滤波参数。"""

    nyquist = config.sample_rate / 2.0
    if config.sample_rate <= 0:
        raise ValueError("fs 必须大于 0")
    if config.order <= 0:
        raise ValueError("Butterworth 阶数必须大于 0")
    if not 0 < config.cutoff < nyquist:
        raise ValueError(f"cutoff 必须位于 0 和 Nyquist 频率 {nyquist} Hz 之间")
    if not 0 < config.band_low < config.band_high < nyquist:
        raise ValueError(
            f"带通范围必须满足 0 < low < high < {nyquist} Hz"
        )
    if config.numtaps < 3 or config.numtaps % 2 == 0:
        raise ValueError("FIR numtaps 必须是大于等于 3 的奇数")
    if config.max_nan_gap < 0:
        raise ValueError("max_nan_gap 不能为负数")


def lowpass_filter(values: np.ndarray, config: FilterConfig) -> np.ndarray:
    """使用数值稳定的 SOS 形式进行零相位 Butterworth 低通滤波。"""

    coefficients = signal.butter(
        config.order,
        config.cutoff,
        btype="lowpass",
        fs=config.sample_rate,
        output="sos",
    )
    return signal.sosfiltfilt(coefficients, values)


def bandpass_filter(values: np.ndarray, config: FilterConfig) -> np.ndarray:
    """使用 Hamming 窗 FIR 系数进行零相位带通滤波。"""

    if len(values) <= 3 * (config.numtaps - 1):
        raise ValueError(
            f"数据只有 {len(values)} 行，无法使用 {config.numtaps} taps 的零相位 FIR；"
            "请减小 --numtaps 或使用更长记录"
        )

    duration = len(values) / config.sample_rate
    if config.band_low < 1.0 / duration:
        warnings.warn(
            f"记录时长仅 {duration:.3f}s，低截止频率 {config.band_low}Hz "
            "低于约一个完整周期，结果可能不可靠。",
            RuntimeWarning,
            stacklevel=2,
        )

    coefficients = signal.firwin(
        config.numtaps,
        (config.band_low, config.band_high),
        pass_zero=False,
        fs=config.sample_rate,
        window="hamming",
    )
    return signal.filtfilt(coefficients, [1.0], values)


def filter_frame(frame: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """仅替换用户选择的列，未选择列保持裁剪后的原值。"""

    result = frame.copy()
    for column_index in config.columns:
        column_name = SENSOR_COLUMNS[column_index]
        clean_values = fill_small_gaps(
            frame.iloc[:, column_index].to_numpy(),
            config.max_nan_gap,
            column_name,
        )
        if config.mode == "lowpass":
            filtered_values = lowpass_filter(clean_values, config)
        else:
            filtered_values = bandpass_filter(clean_values, config)
        result.iloc[:, column_index] = filtered_values
    return result


def write_csv_atomically(frame: pd.DataFrame, output_file: Path) -> None:
    """先写临时文件再替换目标，避免程序中断后留下半个 CSV。"""

    temporary_file = output_file.with_suffix(output_file.suffix + ".tmp")
    frame.to_csv(temporary_file, index=False, header=False, float_format="%.10f")
    temporary_file.replace(output_file)


def save_comparison_plot(
    original: pd.DataFrame,
    filtered: pd.DataFrame,
    source_file: Path,
    output_dir: Path,
    config: FilterConfig,
) -> None:
    """把所选列的滤波前后曲线画在同一张图中，便于肉眼检查。"""

    time_seconds = np.arange(len(original), dtype=float) / config.sample_rate
    figure, axes = plt.subplots(
        len(config.columns),
        1,
        figsize=(12, max(3.2, 2.6 * len(config.columns))),
        sharex=True,
        squeeze=False,
    )
    for row, column_index in enumerate(config.columns):
        axis = axes[row, 0]
        column_name = SENSOR_COLUMNS[column_index]
        axis.plot(time_seconds, original.iloc[:, column_index], alpha=0.45, label="raw")
        axis.plot(time_seconds, filtered.iloc[:, column_index], linewidth=1.2, label="filtered")
        axis.set_ylabel(column_name)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right")
    axes[-1, 0].set_xlabel("Time (s)")
    figure.suptitle(f"{source_file.name} - {config.mode}")
    figure.tight_layout()
    figure.savefig(output_dir / f"{source_file.stem}_{config.mode}.png", dpi=160)
    plt.close(figure)


def process_sensor_file(path: Path, config: FilterConfig) -> Path:
    """读取、裁剪、滤波并保存一个 sensor 文件。"""

    output_file = config.output_dir / f"{path.stem}_{config.mode}_filtered.csv"
    if output_file.exists() and not config.overwrite:
        raise FileExistsError(f"输出已存在；使用 --overwrite 可覆盖：{output_file}")

    raw_frame = read_sensor_csv(path)
    cropped_frame = trim_frame(raw_frame, config.trim_head, config.trim_tail)
    filtered_frame = filter_frame(cropped_frame, config)
    write_csv_atomically(filtered_frame, output_file)
    if config.make_plots:
        save_comparison_plot(cropped_frame, filtered_frame, path, config.output_dir, config)
    return output_file


def build_parser() -> argparse.ArgumentParser:
    """创建命令行接口，并让顶部变量继续作为可编辑默认值。"""

    parser = argparse.ArgumentParser(description="批量滤波原始 6 轴 sensor CSV")
    parser.add_argument("--input-path", type=Path, default=inputpath, help="单个文件或输入文件夹")
    parser.add_argument("--output-path", type=Path, default=outputpath, help="输出文件夹")
    parser.add_argument("--mode", choices=("lowpass", "bandpass"), default=filter_mode)
    parser.add_argument("--fs", type=float, default=fs, help="采样率（Hz）")
    parser.add_argument("--cutoff", type=float, default=lowpass_cutoff, help="低通截止频率（Hz）")
    parser.add_argument("--low", type=float, default=bandpass_low, help="带通低截止频率（Hz）")
    parser.add_argument("--high", type=float, default=bandpass_high, help="带通高截止频率（Hz）")
    parser.add_argument("--order", type=int, default=butterworth_order, help="Butterworth 阶数")
    parser.add_argument("--numtaps", type=int, default=fir_numtaps, help="Hamming FIR 奇数长度")
    parser.add_argument("--columns", default=selected_columns, help="all、TX..FZ 或 0..5，逗号分隔")
    parser.add_argument("--trim-head", type=int, default=trim_head)
    parser.add_argument("--trim-tail", type=int, default=trim_tail)
    parser.add_argument("--max-nan-gap", type=int, default=max_nan_gap)
    parser.add_argument("--plot", action="store_true", help="生成滤波前后对比图")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖已有结果")
    return parser


def main() -> int:
    """构建配置并批量处理；坏文件只计入失败，不中断整个文件夹。"""

    arguments = build_parser().parse_args()
    resolved_input = arguments.input_path.expanduser().resolve()
    resolved_output = (
        arguments.output_path.expanduser().resolve()
        if arguments.output_path is not None
        else (resolved_input.parent if resolved_input.is_file() else resolved_input) / "sensor_filtered"
    )
    resolved_output.mkdir(parents=True, exist_ok=True)

    config = FilterConfig(
        input_path=resolved_input,
        output_dir=resolved_output,
        mode=arguments.mode,
        sample_rate=arguments.fs,
        cutoff=arguments.cutoff,
        band_low=arguments.low,
        band_high=arguments.high,
        order=arguments.order,
        numtaps=arguments.numtaps,
        columns=parse_columns(arguments.columns),
        trim_head=arguments.trim_head,
        trim_tail=arguments.trim_tail,
        max_nan_gap=arguments.max_nan_gap,
        make_plots=arguments.plot,
        overwrite=arguments.overwrite,
    )
    validate_filter_config(config)

    files = discover_sensor_files(config.input_path)
    if not files:
        print(f"未找到合法文件：{config.input_path}/sensor_编号_ang角度.csv")
        return 1

    success_count = 0
    for path in files:
        try:
            output_file = process_sensor_file(path, config)
            success_count += 1
            print(f"[成功] {path.name} -> {output_file}")
        except Exception as error:  # 批处理需要继续，因此在文件粒度捕获异常。
            print(f"[失败] {path.name}：{error}")

    print(f"处理结束：成功 {success_count}/{len(files)} 个文件")
    return 0 if success_count == len(files) else 2


if __name__ == "__main__":
    raise SystemExit(main())
