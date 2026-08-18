import math
import os
from pathlib import Path

import cv2


VIDEO_LIST_FILE = os.environ.get("VIDEO_LIST_FILE", "").strip()
SEGMENT_SECONDS = float(os.environ.get("SEGMENT_SECONDS", "10"))
FOURCC = os.environ.get("VIDEO_FOURCC", "mp4v")
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1").strip() != "0"
MIN_VALID_BYTES = int(os.environ.get("MIN_VALID_BYTES", "1024"))


def load_video_paths() -> list[Path]:
    if not VIDEO_LIST_FILE:
        raise ValueError("VIDEO_LIST_FILE is required")
    with open(VIDEO_LIST_FILE, "r", encoding="utf-8") as f:
        paths = [Path(line.strip()) for line in f if line.strip()]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"missing videos: {missing[:5]}")
    return paths


def open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {path}")
    return writer


def is_valid(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_VALID_BYTES


def split_video(video_path: Path) -> tuple[int, int]:
    chunk_name = video_path.stem
    out_dir = video_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"invalid video metadata: {video_path} fps={fps} size={width}x{height}")

    frames_per_segment = max(1, int(round(fps * SEGMENT_SECONDS)))
    frame_idx = 0
    seg_idx = 0
    writer = None
    written_segments = 0
    skipped_segments = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        target_seg_idx = frame_idx // frames_per_segment
        if target_seg_idx != seg_idx:
            if writer is not None:
                writer.release()
                writer = None
                written_segments += 1
            seg_idx = target_seg_idx

        out_path = out_dir / f"{seg_idx:04d}.mp4"
        if writer is None:
            if SKIP_EXISTING and is_valid(out_path):
                # Skip decoding writes for this whole segment, but keep consuming frames.
                skipped_segments += 1
                writer = False
            else:
                if out_path.exists():
                    out_path.unlink()
                writer = open_writer(out_path, fps, width, height)

        if writer not in (None, False):
            writer.write(frame)

        frame_idx += 1

    if writer not in (None, False):
        writer.release()
        written_segments += 1
    elif writer is False:
        skipped_segments += 1

    cap.release()
    total_segments = int(math.ceil(frame_idx / frames_per_segment)) if frame_idx else 0
    return total_segments, skipped_segments


def main():
    videos = load_video_paths()
    print(f"[INFO] videos={len(videos)}")
    print(f"[INFO] segment_seconds={SEGMENT_SECONDS}")
    print(f"[INFO] fourcc={FOURCC}")
    for idx, video_path in enumerate(videos, 1):
        print(f"[{idx}/{len(videos)}] {video_path}")
        total_segments, skipped_segments = split_video(video_path)
        print(f"  -> segments={total_segments} skipped_existing={skipped_segments}")


if __name__ == "__main__":
    main()
