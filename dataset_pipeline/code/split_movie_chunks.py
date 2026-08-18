#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
DEFAULT_GRID_ROWS = int(os.environ.get("GRID_ROWS", "4"))
DEFAULT_GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
DEFAULT_SHOT_FRAMES = int(os.environ.get("SHOT_FRAMES", "81"))
DEFAULT_CHUNK_FRAMES = int(
    os.environ.get(
        "CHUNK_FRAMES",
        str(DEFAULT_GRID_ROWS * DEFAULT_GRID_COLS * DEFAULT_SHOT_FRAMES),
    )
)
DEFAULT_INDEX_DIGITS = 4


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Split long videos into full-length chunks by frame count, then split "
            "each chunk into fixed-length shot clips."
        )
    )
    parser.add_argument("input_path", help="A long video or a directory of long videos.")
    parser.add_argument("-o", "--output-root", required=True)
    parser.add_argument("--chunk-frames", type=int, default=DEFAULT_CHUNK_FRAMES)
    parser.add_argument("--shot-frames", type=int, default=DEFAULT_SHOT_FRAMES)
    parser.add_argument("--grid-rows", type=int, default=DEFAULT_GRID_ROWS)
    parser.add_argument("--grid-cols", type=int, default=DEFAULT_GRID_COLS)
    parser.add_argument("--chunk-start", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trim-black-boundaries", action="store_true")
    parser.add_argument("--dark-threshold", type=int, default=20)
    parser.add_argument("--dark-ratio", type=float, default=0.97)
    parser.add_argument("--mean-threshold", type=float, default=35.0)
    parser.add_argument("--center-crop-ratio", type=float, default=0.6)
    parser.add_argument("--subtitle-band-ratio", type=float, default=0.22)
    parser.add_argument("--bright-threshold", type=int, default=185)
    parser.add_argument("--subtitle-bright-ratio-min", type=float, default=0.002)
    parser.add_argument("--sparse-bright-ratio-max", type=float, default=0.08)
    parser.add_argument("--std-threshold", type=float, default=32.0)
    return parser


def iter_videos(input_path: Path, recursive: bool) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video extension: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    return sorted(
        path for path in iterator if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def ensure_writer(
    output_path: Path,
    fps: float,
    width: int,
    height: int,
    overwrite: bool,
) -> cv2.VideoWriter:
    if output_path.exists():
        if overwrite:
            output_path.unlink()
        else:
            raise FileExistsError(f"Output file already exists: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")
    return writer


def is_black_like_frame(
    frame,
    dark_threshold: int,
    dark_ratio: float,
    mean_threshold: float,
    center_crop_ratio: float,
    subtitle_band_ratio: float,
    bright_threshold: int,
    subtitle_bright_ratio_min: float,
    sparse_bright_ratio_max: float,
    std_threshold: float,
) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    full_mean = float(gray.mean())
    full_std = float(gray.std())

    if 0 < center_crop_ratio < 1:
        height, width = gray.shape
        crop_height = max(1, int(height * center_crop_ratio))
        crop_width = max(1, int(width * center_crop_ratio))
        top = (height - crop_height) // 2
        left = (width - crop_width) // 2
        center_gray = gray[top:top + crop_height, left:left + crop_width]
    else:
        center_gray = gray

    dark_fraction = (center_gray < dark_threshold).mean()
    mean_value = float(center_gray.mean())
    if (
        dark_fraction >= dark_ratio
        or mean_value <= mean_threshold * 0.35
        or full_mean <= mean_threshold * 0.25
    ):
        return True

    height = gray.shape[0]
    band_height = max(1, int(height * subtitle_band_ratio))
    subtitle_band = gray[height - band_height:, :]
    band_dark_ratio = (subtitle_band < dark_threshold).mean()
    band_bright_ratio = (subtitle_band > bright_threshold).mean()
    full_bright_ratio = (gray > bright_threshold).mean()
    return (
        mean_value <= mean_threshold
        and dark_fraction >= 0.82
        and full_std <= std_threshold
        and subtitle_bright_ratio_min <= full_bright_ratio <= sparse_bright_ratio_max
        and (
            (band_dark_ratio >= 0.55 and band_bright_ratio >= subtitle_bright_ratio_min)
            or full_bright_ratio <= 0.03
        )
    )


def find_valid_frame_range(video_path: Path, total_frames: int, args: argparse.Namespace) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video for boundary scan: {video_path}")
    first_valid = None
    last_valid = None
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not is_black_like_frame(
                frame,
                dark_threshold=args.dark_threshold,
                dark_ratio=args.dark_ratio,
                mean_threshold=args.mean_threshold,
                center_crop_ratio=args.center_crop_ratio,
                subtitle_band_ratio=args.subtitle_band_ratio,
                bright_threshold=args.bright_threshold,
                subtitle_bright_ratio_min=args.subtitle_bright_ratio_min,
                sparse_bright_ratio_max=args.sparse_bright_ratio_max,
                std_threshold=args.std_threshold,
            ):
                if first_valid is None:
                    first_valid = frame_index
                last_valid = frame_index
            frame_index += 1
    finally:
        capture.release()
    if first_valid is None or last_valid is None:
        return 0, total_frames
    return first_valid, last_valid + 1


def write_next_chunk_and_shots(
    capture: cv2.VideoCapture,
    end_frame: int,
    movie_dir: Path,
    base_name: str,
    chunk_index: int,
    fps: float,
    width: int,
    height: int,
    chunk_frames: int,
    shot_frames: int,
    overwrite: bool,
) -> int:
    chunk_name = f"{base_name}_chunk{chunk_index:0{DEFAULT_INDEX_DIGITS}d}"
    chunk_path = movie_dir / f"{chunk_name}.mp4"
    shot_dir = movie_dir / "shot" / chunk_name
    partial_chunk_path = movie_dir / f".{chunk_name}.partial.mp4"
    partial_shot_dir = movie_dir / "shot" / f".{chunk_name}.partial"

    if partial_chunk_path.exists():
        partial_chunk_path.unlink()
    if partial_shot_dir.exists():
        shutil.rmtree(partial_shot_dir)
    partial_shot_dir.mkdir(parents=True, exist_ok=True)

    chunk_writer = ensure_writer(partial_chunk_path, fps, width, height, True)
    shot_writer = None
    written = 0
    try:
        while written < chunk_frames:
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0) >= end_frame:
                break
            ok, frame = capture.read()
            if not ok:
                break
            if written % shot_frames == 0:
                shot_index = written // shot_frames
                shot_path = partial_shot_dir / f"{shot_index:0{DEFAULT_INDEX_DIGITS}d}.mp4"
                shot_writer = ensure_writer(shot_path, fps, width, height, True)
            chunk_writer.write(frame)
            shot_writer.write(frame)
            written += 1
            if written % shot_frames == 0:
                shot_writer.release()
                shot_writer = None
    finally:
        if shot_writer is not None:
            shot_writer.release()
        chunk_writer.release()

    if written < chunk_frames:
        partial_chunk_path.unlink(missing_ok=True)
        shutil.rmtree(partial_shot_dir, ignore_errors=True)
        return written

    if chunk_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output file already exists: {chunk_path}")
        chunk_path.unlink()
    if shot_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {shot_dir}")
        shutil.rmtree(shot_dir)
    partial_shot_dir.replace(shot_dir)
    partial_chunk_path.replace(chunk_path)
    return written


def process_video(video_path: Path, output_root: Path, args: argparse.Namespace) -> None:
    movie_dir = output_root / video_path.stem
    movie_dir.mkdir(parents=True, exist_ok=True)
    existing_chunks = sorted(movie_dir.glob(f"{video_path.stem}_chunk*.mp4"))
    if existing_chunks and not args.overwrite:
        print(f"Skipping {video_path.name}: found {len(existing_chunks)} existing chunk(s)")
        return

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid video metadata: {video_path}")

    start_frame, end_frame = 0, total_frames
    if args.trim_black_boundaries:
        start_frame, end_frame = find_valid_frame_range(video_path, total_frames, args)
        capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(
        f"Processing {video_path.name}: frames={total_frames}, usable={end_frame - start_frame}, "
        f"fps={fps:.3f}, resolution={width}x{height}"
    )
    chunk_index = args.chunk_start
    while True:
        written = write_next_chunk_and_shots(
            capture,
            end_frame,
            movie_dir,
            video_path.stem,
            chunk_index,
            fps,
            width,
            height,
            args.chunk_frames,
            args.shot_frames,
            args.overwrite,
        )
        if written < args.chunk_frames:
            break
        print(
            f"  wrote chunk {chunk_index:0{DEFAULT_INDEX_DIGITS}d}: "
            f"frames={written} shots={written // args.shot_frames}"
        )
        chunk_index += 1
    capture.release()


def main() -> int:
    args = build_parser().parse_args()
    if min(args.chunk_frames, args.shot_frames, args.grid_rows, args.grid_cols) <= 0:
        raise ValueError("Frame counts and grid dimensions must be positive")
    if args.chunk_frames % args.shot_frames != 0:
        raise ValueError("--chunk-frames must be divisible by --shot-frames")
    shots_per_chunk = args.chunk_frames // args.shot_frames
    expected_shots = args.grid_rows * args.grid_cols
    if shots_per_chunk != expected_shots:
        raise ValueError(
            f"Each chunk must contain exactly {expected_shots} shots for one "
            f"{args.grid_rows}x{args.grid_cols} grid; set --chunk-frames to "
            f"{expected_shots} * --shot-frames"
        )

    input_path = Path(args.input_path)
    output_root = Path(args.output_root)
    videos = iter_videos(input_path, args.recursive)
    if len({video.stem for video in videos}) != len(videos):
        raise ValueError("Video filenames must have unique stems")
    if not videos:
        print(f"No videos found in {input_path}")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"Found {len(videos)} video(s); chunk_frames={args.chunk_frames}, "
        f"shot_frames={args.shot_frames}, shots_per_chunk={shots_per_chunk}, "
        f"grid={args.grid_rows}x{args.grid_cols}"
    )
    for video_path in videos:
        process_video(video_path, output_root, args)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
