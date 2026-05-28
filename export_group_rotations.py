import argparse
from pathlib import Path

from generator import get_rotated_groups
from visualize_hex_lattice import (
    build_hex_coordinates,
    build_neighbor_pairs,
    read_input_csv,
    value_to_hex_color,
)


def write_single_sample_svg(
    values: list[float],
    output_path: Path,
    title: str,
    show_ids: bool = False,
) -> None:
    coords = build_hex_coordinates()
    pairs = build_neighbor_pairs(coords)

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    vmin = min(values)
    vmax = max(values)

    scale = 90.0
    pad_x = 62.0
    pad_y = 44.0
    title_h = 34.0
    radius = 18.0

    canvas_w = (max_x - min_x) * scale + 2 * pad_x
    canvas_h = (max_y - min_y) * scale + 2 * pad_y + title_h

    pixel_coords = []
    top_y = title_h
    for x, y in coords:
        px = pad_x + (x - min_x) * scale
        py = top_y + pad_y + (max_y - y) * scale
        pixel_coords.append((px, py))

    elements = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">'
    )
    elements.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    elements.append(
        f'<text x="{canvas_w / 2:.2f}" y="22" text-anchor="middle" '
        f'font-family="Arial, sans-serif" font-size="17" fill="#222">{title}</text>'
    )

    for a, b in pairs:
        x1, y1 = pixel_coords[a]
        x2, y2 = pixel_coords[b]
        elements.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#D0D0D0" stroke-width="2"/>'
        )

    for idx, (px, py) in enumerate(pixel_coords):
        color = value_to_hex_color(values[idx], vmin, vmax)
        elements.append(
            f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" fill="{color}" '
            f'stroke="#2f2f2f" stroke-width="1.5"/>'
        )
        if show_ids:
            elements.append(
                f'<text x="{px:.2f}" y="{py + 4:.2f}" text-anchor="middle" '
                f'font-family="Arial, sans-serif" font-size="11" fill="#ffffff" '
                f'font-weight="bold">{idx}</text>'
            )

    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def export_all_rotations(
    input_csv: Path,
    output_root: Path,
    show_ids: bool = False,
) -> None:
    samples = read_input_csv(input_csv)
    # Sort by root count (number of non-zero entries), ascending.
    samples = sorted(samples, key=lambda row: sum(1 for v in row if float(v) != 0.0))
    output_root.mkdir(parents=True, exist_ok=True)

    for sample_idx, sample in enumerate(samples, start=1):
        # get_rotated_groups expects int list (0/1 in this project).
        base = [int(v) for v in sample]
        rotations = get_rotated_groups(base)  # 0,60,120,180,240,300

        group_dir = output_root / f"group_{sample_idx:03d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        for rotation_idx, rotation_values in enumerate(rotations):
            angle = rotation_idx * 60
            filename = f"rot_{angle:03d}.svg"
            title = f"Group {sample_idx} | Rotate {angle} deg"
            write_single_sample_svg(
                values=rotation_values,
                output_path=group_dir / filename,
                title=title,
                show_ids=show_ids,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Export 6 rotated plots for each 37-point group into separate folders."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="input.csv",
        help="Input CSV path (default: input.csv)",
    )
    parser.add_argument(
        "--output-root",
        "-o",
        default="rotated_plots",
        help="Output root directory (default: rotated_plots)",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Show point IDs 0..36 on dots.",
    )
    args = parser.parse_args()

    input_csv = Path(args.input)
    output_root = Path(args.output_root)
    export_all_rotations(
        input_csv=input_csv,
        output_root=output_root,
        show_ids=args.show_ids,
    )
    print(f"Saved rotated plots under: {output_root.resolve()}")


if __name__ == "__main__":
    main()
