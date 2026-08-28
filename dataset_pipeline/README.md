# MGLV Dataset Pipeline

This directory contains the preprocessing and captioning pipeline used to build MovieGrid 16-grid and 64-grid training samples. Grid geometry is selected through a configuration file. All data paths are supplied through command-line arguments or environment variables; no machine-specific paths are required.

## Installation

```shell
pip install -r dataset_pipeline/requirements.txt
```

FFmpeg must also be available on `PATH`. Set `FFMPEG_EXE` when using a custom FFmpeg installation.

## Grid Configuration

Choose one configuration before running the pipeline:

```shell
CONFIG=dataset_pipeline/configs/16grid.env
# CONFIG=dataset_pipeline/configs/64grid.env
```

| Configuration | Layout | Clips per sample | Frames per clip | Cell size | Grid-video size |
|---|---:|---:|---:|---:|---:|
| `16grid.env` | 4×4 | 16 | 81 | 640×384 | 2560×1536 |
| `64grid.env` | 8×8 | 64 | 81 | 320×192 | 2560×1536 |

`run_with_config.sh` validates the selected configuration and exports it to the requested pipeline command.

## Qwen3-VL Captioning

The caption pipeline uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) in two stages:

1. `total_vlm.py` extracts a global entity catalog and timeline from each concatenated chunk video.
2. `video_cap.py` generates the scene, action, visual style, and caption for each short segment.

The checkpoint is downloaded automatically by Transformers on the first run. Set `MODEL_ID` to use another compatible Qwen3-VL checkpoint.

## Input Video Layout

The initial input is a collection of long videos:

```text
<long_video_root>/
├── movie_0001.mp4
├── movie_0002.mp4
└── ...
```

`split_movie_chunks.py` divides every long video into complete chunks and then divides each chunk into fixed-length shot clips. The selected configuration determines whether a chunk contains 16 or 64 clips:

```text
<split_root>/
└── movie_0001/
    ├── movie_0001_chunk0000.mp4
    └── shot/
        └── movie_0001_chunk0000/
            ├── 0000.mp4
            ├── 0001.mp4
            ├── ...
            └── 0015.mp4    # 16grid
```

For `64grid.env`, the same directory continues through `0063.mp4`. Clips are mapped to the selected grid in natural filename order, from the top-left cell to the bottom-right cell. One chunk corresponds to one grid training sample.

Each clip is resized with aspect-ratio preservation and padding to the configured cell size. Both supplied configurations produce a `2560×1536` grid video.

## Output Layout

The pipeline produces the following files:

```text
<video_root>/
└── movie_0001/
    ├── video/
    │   └── movie_0001_chunk0000.mp4
    └── vlm/
        ├── movie_0001_chunk0000.mp4
        └── movie_0001_chunk0000/
            ├── 0000.mp4
            ├── 0001.mp4
            └── ...

<captions_root>/
└── <case_id>/
    └── vlm/
        └── <chunk_id>.json

<training_root>/
└── <case_id>/
    └── <chunk_id>.caption.txt
```

## 1. Split Long Videos

```shell
bash dataset_pipeline/run_with_config.sh "$CONFIG" \
  python dataset_pipeline/code/split_movie_chunks.py \
    path/to/long_video_root \
    --output-root path/to/split_root
```

`16grid.env` uses `1296 = 16 × 81` frames per chunk. `64grid.env` uses `5184 = 64 × 81`. Frames are written incrementally so a complete 64-grid chunk is not retained in memory. An incomplete tail chunk is discarded. Use `--recursive` for nested input folders and `--trim-black-boundaries` to remove black intros and outros.

## 2. Build Grid and VLM Videos

The runner creates a tiled grid video for training and a temporally concatenated video for VLM captioning.

```shell
ROOT=path/to/split_root \
OUT_ROOT=path/to/video_root \
MAX_PARALLEL_CASES=8 \
bash dataset_pipeline/run_with_config.sh "$CONFIG" \
  bash dataset_pipeline/code/run_grid_videos_parallel.sh
```

Optional variables include `START_CASE_ID`, `END_CASE_ID`, `PYTHON_BIN`, `FFMPEG_EXE`, and `SKIP_EXISTING`.

## 3. Generate Global VLM Captions

Create a text file containing one VLM video path per line, then launch one shard per GPU:

```shell
DATASET_ROOT=path/to/video_root \
bash dataset_pipeline/run_with_config.sh "$CONFIG" \
  bash dataset_pipeline/code/run_total_vlm_shards.sh \
    path/to/video_list.txt \
    path/to/captions_root \
    moviegrid_vlm \
    8
```

The default model is `Qwen/Qwen3-VL-8B-Instruct`. Override it with `MODEL_ID`.

## 4. Split VLM Videos into Segments

```shell
VIDEO_LIST_FILE=path/to/video_list.txt \
bash dataset_pipeline/run_with_config.sh "$CONFIG" \
  python dataset_pipeline/code/make_vlm_clips_cv2.py
```

Segments are ten seconds by default. Set `SEGMENT_SECONDS` to change the duration.

## 5. Generate Per-Segment Captions

```shell
DATASET_ROOT=path/to/video_root \
CAPTIONS_ROOT=path/to/captions_root \
bash dataset_pipeline/run_with_config.sh "$CONFIG" \
  python dataset_pipeline/code/video_cap.py
```

Multi-GPU execution is available through `SHARD_COUNT`, `SHARD_INDEX`, and `CUDA_VISIBLE_DEVICES`.

## 6. Build Training Captions

```shell
python dataset_pipeline/code/build_chunk_captions.py \
  --src-root path/to/captions_root \
  --dst-root path/to/training_root
```

`build_merged_captions.py` can additionally write merged captions back to the JSON files:

```shell
python dataset_pipeline/code/build_merged_captions.py \
  --captions-root path/to/captions_root
```
