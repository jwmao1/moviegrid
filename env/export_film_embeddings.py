#!/usr/bin/env python3
import argparse
import importlib
import importlib.util
import json
import os
import sys
import tomllib
import types
from pathlib import Path

import imageio.v3 as iio
import torch
from tqdm import tqdm


SNAPSHOT_ROOT = Path(__file__).resolve().parents[1]
DIFFUSION_PIPE_ROOT = SNAPSHOT_ROOT / "diffusion-pipe"
sys.path.insert(0, str(SNAPSHOT_ROOT))
sys.path.insert(0, str(DIFFUSION_PIPE_ROOT))

models_pkg = types.ModuleType("models")
models_pkg.__path__ = [str(DIFFUSION_PIPE_ROOT / "models")]
sys.modules["models"] = models_pkg

utils_pkg = types.ModuleType("utils")
utils_pkg.__path__ = [str(DIFFUSION_PIPE_ROOT / "utils")]
sys.modules["utils"] = utils_pkg
common_spec = importlib.util.spec_from_file_location(
    "utils.common", DIFFUSION_PIPE_ROOT / "utils" / "common.py"
)
common_module = importlib.util.module_from_spec(common_spec)
sys.modules["utils.common"] = common_module
common_spec.loader.exec_module(common_module)

base_spec = importlib.util.spec_from_file_location(
    "models.base", DIFFUSION_PIPE_ROOT / "models" / "base.py"
)
base_module = importlib.util.module_from_spec(base_spec)
sys.modules["models.base"] = base_module
base_spec.loader.exec_module(base_module)

from models.wan.wan import WanPipeline  # noqa: E402


DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def parse_dtype(value: str) -> torch.dtype:
    normalized = DTYPE_MAP.get(value.lower())
    if normalized is None:
        raise ValueError(f"Unsupported dtype: {value}")
    return normalized


def load_config(config_path: Path) -> dict:
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def normalize_config_dtypes(config: dict) -> dict:
    config = dict(config)
    model_config = dict(config.get("model", {}))
    for key in ("dtype", "transformer_dtype"):
        value = model_config.get(key)
        if isinstance(value, str):
            normalized = DTYPE_MAP.get(value.lower())
            if normalized is None:
                raise ValueError(f"Unsupported dtype string for model.{key}: {value}")
            model_config[key] = normalized
    config["model"] = model_config
    return config


def list_pairs(dataset_root: Path):
    mp4_files = sorted(dataset_root.rglob("*.mp4"))
    pairs = []
    for mp4_path in mp4_files:
        txt_path = mp4_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append((mp4_path, txt_path))
    return pairs


def count_source_frames(video_path: Path, fps: int) -> int:
    count = 0
    for _ in iio.imiter(str(video_path), fps=fps):
        count += 1
    return count


def triplet_exists(stem_path: Path) -> bool:
    return (
        stem_path.with_suffix(".meta.json").exists()
        and stem_path.with_suffix(".text_emb.pt").exists()
        and stem_path.with_suffix(".video_emb.pt").exists()
    )


def write_skip(skip_path: Path, key: str, mp4_path: Path, txt_path: Path, reason: str, detail: str = ""):
    skip_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key": key,
        "txt": str(txt_path),
        "mp4": str(mp4_path),
        "reason": reason,
    }
    if detail:
        payload["detail"] = detail
    tmp_path = skip_path.with_suffix(skip_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(skip_path)


def resolve_dataset_spec(config_path: Path, config: dict):
    dataset_cfg = config.get("dataset")
    if not dataset_cfg:
        return (2560, 1536), [101]
    dataset_path = Path(dataset_cfg)
    if not dataset_path.is_absolute():
        dataset_path = (config_path.parent / dataset_path).resolve()
    with open(dataset_path, "rb") as f:
        dataset_config = tomllib.load(f)
    resolutions = dataset_config.get("resolutions", [[2560, 1536]])
    resolution = tuple(int(v) for v in resolutions[0])
    frame_buckets = sorted(int(v) for v in dataset_config.get("frame_buckets", [101]) if int(v) > 1)
    if not frame_buckets:
        frame_buckets = [101]
    return resolution, frame_buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--only-key",
        default="",
        help="Optional dataset-relative stem, e.g. 154/154_chunk0000, to export a single pair.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--save-dtype",
        default="float32",
        choices=sorted(DTYPE_MAP.keys()),
        help="Floating dtype used for saved text/video embedding tensors.",
    )
    args = ap.parse_args()

    config_path = Path(args.config).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dtype = parse_dtype(args.save_dtype)

    config = normalize_config_dtypes(load_config(config_path))
    config.setdefault("model", {})
    # Keep the full text sequence available to the tokenizer.
    config["model"]["tokenizer_max_length"] = 0
    target_resolution, frame_buckets = resolve_dataset_spec(config_path, config)

    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    pipe = WanPipeline(config)
    preprocess_media = pipe.get_preprocess_media_file_fn()
    vae = pipe.get_vae().to(device)
    call_vae = pipe.get_call_vae_fn(vae)
    text_encoder = pipe.get_text_encoders()[0].to(device)
    tokenizer = pipe.text_encoder.tokenizer
    prompt_prefix = pipe.prompt_prefix_text.strip()

    pairs = list_pairs(dataset_root)
    if args.only_key:
        only_key = args.only_key.strip().strip("/")
        pairs = [
            (mp4_path, txt_path)
            for mp4_path, txt_path in pairs
            if mp4_path.relative_to(dataset_root).with_suffix("").as_posix() == only_key
        ]
        if not pairs:
            raise ValueError(f"No dataset pair found for --only-key {args.only_key!r}")
    shard_pairs = pairs[args.shard_index::args.num_shards]
    if args.limit > 0:
        shard_pairs = shard_pairs[:args.limit]

    print(
        f"[export] dataset_root={dataset_root} output_dir={output_dir} "
        f"pairs={len(pairs)} shard={args.shard_index}/{args.num_shards} shard_pairs={len(shard_pairs)} "
        f"resolution={target_resolution} frame_buckets={frame_buckets} save_dtype={args.save_dtype}",
        flush=True,
    )

    for mp4_path, txt_path in tqdm(shard_pairs):
        rel_stem = mp4_path.relative_to(dataset_root).with_suffix("")
        key = rel_stem.as_posix()
        stem_path = output_dir / rel_stem
        skip_path = stem_path.with_suffix(".skip.json")

        if (not args.overwrite) and (triplet_exists(stem_path) or skip_path.exists()):
            continue

        try:
            source_frames = count_source_frames(mp4_path, pipe.framerate)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            write_skip(skip_path, key, mp4_path, txt_path, "unreadable_video", detail)
            print(f"[skip] unreadable video: {mp4_path} error={detail}", flush=True)
            continue

        valid_buckets = [bucket for bucket in frame_buckets if source_frames >= bucket]
        if not valid_buckets:
            write_skip(
                skip_path,
                key,
                mp4_path,
                txt_path,
                "source_video_too_short",
                f"source_frames={source_frames} frame_buckets={frame_buckets}",
            )
            print(
                f"[skip] source video too short for configured frame buckets: "
                f"{mp4_path} source_frames={source_frames} frame_buckets={frame_buckets}",
                flush=True,
            )
            continue
        target_frames = valid_buckets[-1]
        try:
            clips = preprocess_media(
                (None, str(mp4_path)),
                None,
                (target_resolution[0], target_resolution[1], target_frames),
            )
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            write_skip(skip_path, key, mp4_path, txt_path, "preprocess_failed", detail)
            print(f"[skip] preprocess failed for {mp4_path} error={detail}", flush=True)
            continue
        if len(clips) == 0:
            write_skip(skip_path, key, mp4_path, txt_path, "no_clips_extracted", f"target_frames={target_frames}")
            print(f"[skip] no clips extracted for {mp4_path} with target_frames={target_frames}", flush=True)
            continue

        caption_text = txt_path.read_text(encoding="utf-8").strip()
        full_caption = f"{prompt_prefix} {caption_text}".strip() if prompt_prefix else caption_text

        ids, mask = tokenizer([full_caption], return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_len = int(mask.gt(0).sum(dim=1)[0].item())
        with torch.autocast(device_type="cuda", dtype=next(text_encoder.parameters()).dtype):
            text_context = text_encoder(ids, mask)[0, :seq_len].detach().cpu().to(save_dtype)
        input_ids = ids[0].detach().cpu()
        attention_mask = mask[0].detach().cpu().to(torch.int8)

        for clip_idx, (video_tensor, _) in enumerate(clips):
            key = rel_stem.as_posix() if len(clips) == 1 else f"{rel_stem.as_posix()}_clip{clip_idx:04d}"
            stem_path = output_dir / key
            meta_path = stem_path.with_suffix(".meta.json")
            video_path = stem_path.with_suffix(".video_emb.pt")
            text_path = stem_path.with_suffix(".text_emb.pt")

            if (not args.overwrite) and meta_path.exists() and video_path.exists() and text_path.exists():
                continue

            stem_path.parent.mkdir(parents=True, exist_ok=True)
            if skip_path.exists():
                skip_path.unlink()

            tensor = video_tensor.unsqueeze(0)
            vae_result = call_vae(tensor)
            latents = vae_result["latents"][0].detach().cpu().to(save_dtype)
            sampled_frames = int(video_tensor.shape[1])
            aligned_hw = [int(video_tensor.shape[2]), int(video_tensor.shape[3])]

            torch.save(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "seq_len": seq_len,
                    "max_text_tokens": int(input_ids.shape[0]),
                    "raw_token_count": seq_len,
                    "truncated": False,
                    "text_context": text_context,
                    "context_dim": int(text_context.shape[-1]),
                    "dtype": str(text_context.dtype).replace("torch.", ""),
                },
                text_path,
            )
            torch.save(
                {
                    "video_latent": latents,
                    "latent_shape": list(latents.shape),
                    "dtype": str(latents.dtype).replace("torch.", ""),
                },
                video_path,
            )
            meta = {
                "key": key,
                "txt": str(txt_path),
                "mp4": str(mp4_path),
                "text_vec": str(text_path),
                "video_vec": str(video_path),
                "raw_token_count": seq_len,
                "seq_len": seq_len,
                "truncated": False,
                "video_latent_shape": list(latents.shape),
                "embedding_dtype": str(save_dtype).replace("torch.", ""),
                "video_stats": {
                    "source_fps": float(pipe.framerate),
                    "sampled_frames": sampled_frames,
                    "source_frames": source_frames,
                    "sample_indices": list(range(sampled_frames)),
                    "aligned_hw": aligned_hw,
                },
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"[saved] key={key} source_frames={source_frames} sampled_frames={sampled_frames} "
                f"seq_len={seq_len} latent_shape={list(latents.shape)}",
                flush=True,
            )

    print("[export] complete", flush=True)


if __name__ == "__main__":
    main()
