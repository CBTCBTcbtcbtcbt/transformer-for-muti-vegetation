import argparse
import csv
import math
from pathlib import Path


ROW_LENGTHS = [4, 5, 6, 7, 6, 5, 4]
TOTAL_POINTS = sum(ROW_LENGTHS)  # 37


def build_hex_coordinates():
    coords = []
    for r, row_len in enumerate(ROW_LENGTHS):
        x_start = -(row_len - 1) / 2.0
        y = -r * (math.sqrt(3) / 2.0)
        for c in range(row_len):
            x = x_start + c
            coords.append((x, y))
    return coords


def read_input_csv(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            cleaned = [cell.strip() for cell in raw_row if cell.strip() != ""]
            if not cleaned:
                continue
            try:
                values = [float(cell) for cell in cleaned]
            except ValueError:
                continue

            if len(values) == TOTAL_POINTS:
                rows.append(values)
            elif len(values) == 1:
                rows.append(values)
            else:
                raise ValueError(
                    f"Found a row with length {len(values)}; expected {TOTAL_POINTS} values."
                )

    if not rows:
        raise ValueError("No valid numeric rows found in input CSV.")

    # Support single-column 37-row format.
    if all(len(r) == 1 for r in rows):
        if len(rows) != TOTAL_POINTS:
            raise ValueError(
                f"Single-column input detected, but row count is {len(rows)}; expected {TOTAL_POINTS}."
            )
        return [[r[0] for r in rows]]

    if any(len(r) != TOTAL_POINTS for r in rows):
        raise ValueError("Inconsistent row lengths in CSV.")

    return rows


def build_neighbor_pairs(coords):
    threshold = 1.001
    pairs = []
    n = len(coords)
    for i in range(n):
        x1, y1 = coords[i]
        for j in range(i + 1, n):
            x2, y2 = coords[j]
            if math.hypot(x1 - x2, y1 - y2) <= threshold:
                pairs.append((i, j))
    return pairs


def lerp(a, b, t):
    return a + (b - a) * t


def value_to_hex_color(v, vmin, vmax):
    # Viridis-like stops
    stops = [
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37),
    ]
    if vmax <= vmin:
        t = 0.5
    else:
        t = (v - vmin) / (vmax - vmin)
        t = max(0.0, min(1.0, t))

    segment = t * (len(stops) - 1)
    i = int(math.floor(segment))
    if i >= len(stops) - 1:
        i = len(stops) - 2
    local_t = segment - i

    r = int(round(lerp(stops[i][0], stops[i + 1][0], local_t)))
    g = int(round(lerp(stops[i][1], stops[i + 1][1], local_t)))
    b = int(round(lerp(stops[i][2], stops[i + 1][2], local_t)))
    return f"#{r:02x}{g:02x}{b:02x}"


def visualize_svg(samples, output_path, show_ids=False, columns=4):
    coords = build_hex_coordinates()
    pairs = build_neighbor_pairs(coords)
    sample_count = len(samples)

    vmin = min(min(s) for s in samples)
    vmax = max(max(s) for s in samples)

    xs = [p[0] for p in coords]
    ys = [p[1] for p in coords]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    scale = 86.0
    pad_x = 58.0
    pad_y = 42.0
    title_h = 28.0
    radius = 18.0

    tile_w = (max_x - min_x) * scale + 2 * pad_x
    tile_h = (max_y - min_y) * scale + 2 * pad_y + title_h

    cols = max(1, columns)
    rows = math.ceil(sample_count / cols)
    canvas_w = cols * tile_w
    canvas_h = rows * tile_h

    elements = []
    elements.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_w:.0f}" height="{canvas_h:.0f}" '
        f'viewBox="0 0 {canvas_w:.0f} {canvas_h:.0f}">'
    )
    elements.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')

    for i, values in enumerate(samples):
        col_idx = i % cols
        row_idx = i // cols
        ox = col_idx * tile_w
        oy = row_idx * tile_h
        top_y = oy + title_h

        elements.append(f'<g id="sample-{i}">')
        elements.append(
            f'<text x="{ox + tile_w / 2:.2f}" y="{oy + 20:.2f}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="16" fill="#222">Sample {i}</text>'
        )

        pixel_coords = []
        for x, y in coords:
            px = ox + pad_x + (x - min_x) * scale
            py = top_y + pad_y + (max_y - y) * scale
            pixel_coords.append((px, py))

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

        elements.append("</g>")

    elements.append("</svg>")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize 37-point hex lattice samples from CSV."
    )
    parser.add_argument(
        "--input", "-i", default="input.csv", help="Input CSV path (default: input.csv)"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="hex_lattice.svg",
        help="Output image path (SVG only, default: hex_lattice.svg)",
    )
    parser.add_argument(
        "--row",
        type=int,
        default=None,
        help="Visualize one sample row only (0-based index).",
    )
    parser.add_argument(
        "--show-ids",
        action="store_true",
        help="Show point IDs (0..36) on each dot.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=4,
        help="Number of subplots per row for multi-sample input (default: 4).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    samples = read_input_csv(input_path)

    if args.row is not None:
        if args.row < 0 or args.row >= len(samples):
            raise IndexError(
                f"--row out of range. Got {args.row}; valid range is 0..{len(samples)-1}."
            )
        samples = [samples[args.row]]

    output_path = Path(args.output)
    if output_path.suffix.lower() != ".svg":
        output_path = output_path.with_suffix(".svg")
        print(f"Changed output extension to SVG: {output_path.name}")

    visualize_svg(
        samples=samples,
        output_path=output_path,
        show_ids=args.show_ids,
        columns=args.columns,
    )
    print(f"Saved visualization to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
