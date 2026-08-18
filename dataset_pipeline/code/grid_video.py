import os
import shutil
import subprocess
import tempfile


VIDEO_CODEC = os.environ.get("VIDEO_CODEC", "mpeg4")
VIDEO_CRF = os.environ.get("VIDEO_CRF", "18")
VIDEO_PRESET = os.environ.get("VIDEO_PRESET", "veryfast")
VIDEO_QUALITY = os.environ.get("VIDEO_QUALITY", "2")
FFMPEG_EXE = os.environ.get("FFMPEG_EXE") or shutil.which("ffmpeg") or "ffmpeg"

GRID_W = int(os.environ.get("GRID_COLS", "4"))
GRID_H = int(os.environ.get("GRID_ROWS", "4"))
CELL_W = int(os.environ.get("CELL_WIDTH", "640"))
CELL_H = int(os.environ.get("CELL_HEIGHT", "384"))
if min(GRID_W, GRID_H, CELL_W, CELL_H) <= 0:
    raise ValueError("Grid dimensions and cell dimensions must be positive")
OUT_W, OUT_H = GRID_W * CELL_W, GRID_H * CELL_H
TILE_N = GRID_W * GRID_H
VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


def video_encode_args() -> list[str]:
    if VIDEO_CODEC == "libx264":
        return [
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", VIDEO_CRF,
            "-preset", VIDEO_PRESET,
        ]
    if VIDEO_CODEC == "libopenh264":
        return [
            "-c:v", "libopenh264",
            "-pix_fmt", "yuv420p",
            "-profile:v", "high",
            "-rc_mode", "quality",
        ]
    if VIDEO_CODEC == "mpeg4":
        return [
            "-c:v", "mpeg4",
            "-pix_fmt", "yuv420p",
            "-q:v", VIDEO_QUALITY,
        ]
    raise ValueError(f"Unsupported VIDEO_CODEC: {VIDEO_CODEC}")


def build_layout(n_inputs: int) -> str:
    return "|".join(
        f"{(index % GRID_W) * CELL_W}_{(index // GRID_W) * CELL_H}"
        for index in range(n_inputs)
    )


def write_filter_script(path: str, n_inputs: int) -> None:
    lines = []
    for index in range(n_inputs):
        lines.append(
            f"[{index}:v]"
            f"scale={CELL_W}:{CELL_H}:force_original_aspect_ratio=decrease,"
            f"pad={CELL_W}:{CELL_H}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"setsar=1[v{index}]"
        )
    lines.append(
        "".join(f"[v{index}]" for index in range(n_inputs))
        + f"xstack=inputs={n_inputs}:layout={build_layout(n_inputs)}:fill=black:shortest=1[vout]"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(";\n".join(lines))


def run_one_mosaic(video_paths: list[str], out_path: str) -> None:
    videos = video_paths[:TILE_N]
    padding_count = TILE_N - len(videos)
    command = [FFMPEG_EXE, "-y", "-hide_banner", "-loglevel", "error"]
    for video_path in videos:
        command += ["-i", video_path]
    for _ in range(padding_count):
        command += [
            "-f", "lavfi",
            "-i", f"color=c=black:s={CELL_W}x{CELL_H}:r=30:d=36000",
        ]

    with tempfile.TemporaryDirectory() as temp_dir:
        script_path = os.path.join(temp_dir, "filter.txt")
        write_filter_script(script_path, TILE_N)
        command += [
            "-filter_complex_script", script_path,
            "-map", "[vout]",
            "-s", f"{OUT_W}x{OUT_H}",
            "-r", "30",
            *video_encode_args(),
            "-an",
            out_path,
        ]
        subprocess.run(command, check=True)
