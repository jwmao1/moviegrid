import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import grid_video


ROOT_VALUE = os.environ.get("ROOT", "").strip()
OUT_ROOT_VALUE = os.environ.get("OUT_ROOT", "").strip()
ROOT = Path(ROOT_VALUE) if ROOT_VALUE else None
OUT_ROOT = Path(OUT_ROOT_VALUE) if OUT_ROOT_VALUE else None
START_CASE_ID = os.environ.get("START_CASE_ID", "").strip() or None
END_CASE_ID = os.environ.get("END_CASE_ID", "").strip() or None
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "1") != "0"
MIN_VALID_BYTES = int(os.environ.get("MIN_VALID_BYTES", "1024"))
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", str(os.cpu_count() or 1)))
FFMPEG_EXE = os.environ.get("FFMPEG_EXE") or grid_video.FFMPEG_EXE

VIDEO_EXTS = grid_video.VIDEO_EXTS


def case_sort_key(path: Path):
    if path.name.isdigit():
        return (0, int(path.name))
    return (1, path.name)


def natural_sort_key(path: Path):
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    return (1, stem, path.name)


def in_requested_range(case_dir: Path) -> bool:
    if not case_dir.name.isdigit():
        return True
    case_id = int(case_dir.name)
    if START_CASE_ID is not None and case_id < int(START_CASE_ID):
        return False
    if END_CASE_ID is not None and case_id > int(END_CASE_ID):
        return False
    return True


def is_valid_output(path: Path) -> bool:
    return path.is_file() and path.stat().st_size >= MIN_VALID_BYTES


def list_chunk_dirs(case_dir: Path) -> list[Path]:
    shot_root = case_dir / "shot"
    chunk_root = shot_root if shot_root.is_dir() else case_dir
    return sorted(
        [p for p in chunk_root.iterdir() if p.is_dir()],
        key=case_sort_key,
    )


def list_chunk_videos(chunk_dir: Path) -> list[Path]:
    videos = []
    for ext in VIDEO_EXTS:
        videos.extend(chunk_dir.glob(f"*{ext}"))
    return sorted(set(videos), key=natural_sort_key)


def concat_one_sequence(video_paths: list[Path], out_path: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        list_path = Path(td) / "concat.txt"
        with open(list_path, "w", encoding="utf-8") as f:
            for vp in video_paths:
                f.write(f"file '{vp.as_posix()}'\n")

        cmd = [
            FFMPEG_EXE,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            *grid_video.video_encode_args(),
            "-an",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)


def process_case(case_dir_str: str) -> list[str]:
    case_dir = Path(case_dir_str)
    out_case_dir = OUT_ROOT / case_dir.name
    out_video_dir = out_case_dir / "video"
    out_vlm_dir = out_case_dir / "vlm"
    out_video_dir.mkdir(parents=True, exist_ok=True)
    out_vlm_dir.mkdir(parents=True, exist_ok=True)

    logs = []
    for chunk_dir in list_chunk_dirs(case_dir):
        clip_paths = list_chunk_videos(chunk_dir)
        if not clip_paths:
            logs.append(f"[SKIP] empty chunk: {chunk_dir}")
            continue

        if len(clip_paths) != grid_video.TILE_N:
            logs.append(
                f"[SKIP] {chunk_dir}: expected {grid_video.TILE_N} clips, "
                f"found {len(clip_paths)}"
            )
            continue

        sample_name = chunk_dir.name
        video_out = out_video_dir / f"{sample_name}.mp4"
        vlm_out = out_vlm_dir / f"{sample_name}.mp4"

        if SKIP_EXISTING and is_valid_output(video_out) and is_valid_output(vlm_out):
            logs.append(f"[SKIP] {case_dir.name}/{sample_name}")
            continue

        if video_out.exists() and not is_valid_output(video_out):
            video_out.unlink()
        if vlm_out.exists() and not is_valid_output(vlm_out):
            vlm_out.unlink()

        if not (SKIP_EXISTING and is_valid_output(video_out)):
            grid_video.run_one_mosaic([str(path) for path in clip_paths], str(video_out))
        if not (SKIP_EXISTING and is_valid_output(vlm_out)):
            concat_one_sequence(clip_paths, vlm_out)

        logs.append(f"[DONE] {case_dir.name}/{sample_name} clips={len(clip_paths)}")

    return logs


def iter_cases(root: Path):
    for case_dir in sorted(root.iterdir(), key=case_sort_key):
        if case_dir.is_dir() and in_requested_range(case_dir):
            yield case_dir


def main():
    if ROOT is None:
        raise ValueError("ROOT is required")
    if OUT_ROOT is None:
        raise ValueError("OUT_ROOT is required")
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT}")
    if shutil.which(str(FFMPEG_EXE)) is None and not os.path.isfile(str(FFMPEG_EXE)):
        raise FileNotFoundError(
            "ffmpeg not found. Install FFmpeg and add it to PATH, "
            "or set FFMPEG_EXE to full path."
        )
    if START_CASE_ID is not None and not START_CASE_ID.isdigit():
        raise ValueError(f"START_CASE_ID must be numeric, got: {START_CASE_ID}")
    if END_CASE_ID is not None and not END_CASE_ID.isdigit():
        raise ValueError(f"END_CASE_ID must be numeric, got: {END_CASE_ID}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    cases = list(iter_cases(ROOT))
    total_chunks = sum(len(list_chunk_dirs(case_dir)) for case_dir in cases)

    print(f"[INFO] root={ROOT}")
    print(f"[INFO] out_root={OUT_ROOT}")
    print(f"[INFO] ffmpeg={FFMPEG_EXE}")
    print(
        f"[INFO] grid={grid_video.GRID_H}x{grid_video.GRID_W} "
        f"cell={grid_video.CELL_W}x{grid_video.CELL_H} "
        f"output={grid_video.OUT_W}x{grid_video.OUT_H} clips={grid_video.TILE_N}"
    )
    print(f"[INFO] start_case_id={START_CASE_ID}")
    print(f"[INFO] end_case_id={END_CASE_ID}")
    print(f"[INFO] skip_existing={SKIP_EXISTING}")
    print(f"[INFO] max_workers={max(1, MAX_WORKERS)}")
    print(f"[INFO] total_cases={len(cases)}")
    print(f"[INFO] total_chunks={total_chunks}")

    with ProcessPoolExecutor(max_workers=max(1, MAX_WORKERS)) as executor:
        futures = [executor.submit(process_case, str(case_dir)) for case_dir in cases]
        for future in as_completed(futures):
            for line in future.result():
                print(line, flush=True)


if __name__ == "__main__":
    main()
