# Panel Sequence Video

`panel_sequence_video.py` crops a regular grid video into individual panels and concatenates them into a single sequential video. The default configuration follows the 16-slot pipeline and uses a `4 x 4` grid.

## Usage

```bash
python utils/panel_sequence_video.py /path/to/sample_preview.mp4
```

By default, the output is written to `<input_stem>_panels_sequential.mp4` in the input video's directory. The options can also be specified explicitly:

```bash
python utils/panel_sequence_video.py /path/to/sample_preview.mp4 \
  --output /path/to/output_panels.mp4 \
  --rows 4 \
  --cols 4 \
  --order row-major \
  --fps 16
```

Supported traversal orders:

- `row-major`: left to right, then top to bottom; this is the default.
- `col-major`: top to bottom, then left to right.
- `snake`: left to right on odd-numbered rows and right to left on even-numbered rows.

The input width must be divisible by `cols`, and the input height must be divisible by `rows`. For a `2560 x 1536` video arranged as a 4×4 grid, each panel is `640 x 384`. An input containing 101 frames produces `101 x 16 = 1616` output frames.

The script depends on `imageio` and `imageio-ffmpeg` and does not require a GPU. It performs fixed geometric cropping without detecting visual content or irregular boundaries.
