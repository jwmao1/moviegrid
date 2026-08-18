#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio


def panel_order(rows: int, cols: int, mode: str) -> list[tuple[int, int]]:
    if mode == "row-major":
        return [(r, c) for r in range(rows) for c in range(cols)]
    if mode == "col-major":
        return [(r, c) for c in range(cols) for r in range(rows)]
    if mode == "snake":
        order: list[tuple[int, int]] = []
        for r in range(rows):
            cols_iter = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
            for c in cols_iter:
                order.append((r, c))
        return order
    raise ValueError(f"unsupported mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crop an RxC grid video into panel clips and concatenate them into one video."
    )
    parser.add_argument("input", type=Path, help="Input grid video")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Default: <input stem>_panels_sequential.mp4",
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument(
        "--order",
        choices=["row-major", "col-major", "snake"],
        default="row-major",
        help="Panel traversal order",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override output fps")
    args = parser.parse_args()

    src = args.input
    out = args.output or src.with_name(f"{src.stem}_panels_sequential.mp4")

    reader = imageio.get_reader(str(src))
    meta = reader.get_meta_data()
    fps = args.fps or meta.get("fps", 16)
    frames = [frame for frame in reader]
    reader.close()
    if not frames:
        raise RuntimeError(f"no frames read from {src}")

    h, w = frames[0].shape[:2]
    if h % args.rows != 0 or w % args.cols != 0:
        raise ValueError(
            f"video size {(w, h)} is not divisible by grid {args.cols}x{args.rows}"
        )
    cell_w = w // args.cols
    cell_h = h // args.rows

    order = panel_order(args.rows, args.cols, args.order)
    writer = imageio.get_writer(
        str(out), fps=fps, codec="libx264", quality=8, macro_block_size=None
    )
    for r, c in order:
        y0, y1 = r * cell_h, (r + 1) * cell_h
        x0, x1 = c * cell_w, (c + 1) * cell_w
        for frame in frames:
            writer.append_data(frame[y0:y1, x0:x1])
    writer.close()

    print(out)
    print(
        {
            "order": args.order,
            "source_wh": (w, h),
            "cell_wh": (cell_w, cell_h),
            "frames_in": len(frames),
            "frames_out": len(frames) * len(order),
            "fps": fps,
        }
    )


if __name__ == "__main__":
    main()
