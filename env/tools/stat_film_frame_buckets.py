#!/usr/bin/env python3
import argparse
import csv
import json
import sys
import time
import tomllib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import imageio.v3 as iio


def list_pairs(dataset_root: Path):
    mp4_files = sorted(dataset_root.rglob("*.mp4"))
    pairs = []
    missing_txt = []
    for mp4_path in mp4_files:
        txt_path = mp4_path.with_suffix(".txt")
        if txt_path.exists():
            pairs.append(mp4_path)
        else:
            missing_txt.append(mp4_path)
    return pairs, missing_txt


def load_frame_buckets(dataset_config: Path):
    with dataset_config.open("rb") as f:
        cfg = tomllib.load(f)
    buckets = sorted(int(v) for v in cfg.get("frame_buckets", [101]) if int(v) > 1)
    return buckets or [101]


def count_source_frames(video_path: str, fps: int):
    # Match export_film_embeddings.py: count frames yielded by imageio at the model framerate.
    count = 0
    for _ in iio.imiter(video_path, fps=fps):
        count += 1
    return count


def assign_bucket(source_frames: int, frame_buckets):
    valid = [bucket for bucket in frame_buckets if source_frames >= bucket]
    return valid[-1] if valid else None


def stat_one(args):
    mp4_path, dataset_root, fps, frame_buckets = args
    mp4_path = Path(mp4_path)
    started = time.time()
    try:
        source_frames = count_source_frames(str(mp4_path), fps)
        bucket = assign_bucket(source_frames, frame_buckets)
        return {
            "ok": True,
            "key": mp4_path.relative_to(dataset_root).with_suffix("").as_posix(),
            "mp4": str(mp4_path),
            "txt": str(mp4_path.with_suffix(".txt")),
            "source_frames": source_frames,
            "bucket": bucket,
            "too_short": bucket is None,
            "elapsed_sec": round(time.time() - started, 4),
        }
    except Exception as exc:
        return {
            "ok": False,
            "key": mp4_path.relative_to(dataset_root).with_suffix("").as_posix(),
            "mp4": str(mp4_path),
            "txt": str(mp4_path.with_suffix(".txt")),
            "error": repr(exc),
            "elapsed_sec": round(time.time() - started, 4),
        }


def write_outputs(out_dir: Path, results, missing_txt, dataset_root: Path, frame_buckets, fps: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "frame_bucket_rows.csv"
    jsonl_path = out_dir / "frame_bucket_rows.jsonl"
    summary_path = out_dir / "summary.json"
    errors_path = out_dir / "errors.jsonl"
    missing_path = out_dir / "missing_txt.jsonl"

    ok_rows = [r for r in results if r.get("ok")]
    error_rows = [r for r in results if not r.get("ok")]
    bucket_counts = Counter(str(r["bucket"]) if r["bucket"] is not None else "too_short" for r in ok_rows)
    source_frame_counts = Counter(str(r["source_frames"]) for r in ok_rows)

    with rows_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["key", "mp4", "txt", "source_frames", "bucket", "too_short", "elapsed_sec"],
        )
        writer.writeheader()
        for row in ok_rows:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in ok_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with errors_path.open("w", encoding="utf-8") as f:
        for row in error_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with missing_path.open("w", encoding="utf-8") as f:
        for path in missing_txt:
            f.write(json.dumps({"mp4": str(path)}, ensure_ascii=False) + "\n")

    frame_values = [r["source_frames"] for r in ok_rows]
    summary = {
        "dataset_root": str(dataset_root),
        "fps": fps,
        "frame_buckets": frame_buckets,
        "total_mp4_with_txt": len(results),
        "ok": len(ok_rows),
        "errors": len(error_rows),
        "missing_txt": len(missing_txt),
        "bucket_counts": dict(sorted(bucket_counts.items(), key=lambda kv: (kv[0] == "too_short", int(kv[0]) if kv[0].isdigit() else -1))),
        "source_frame_counts": dict(sorted(source_frame_counts.items(), key=lambda kv: int(kv[0]))),
        "min_source_frames": min(frame_values) if frame_values else None,
        "max_source_frames": max(frame_values) if frame_values else None,
        "rows_csv": str(rows_path),
        "rows_jsonl": str(jsonl_path),
        "errors_jsonl": str(errors_path),
        "missing_txt_jsonl": str(missing_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    dataset_config = Path(args.dataset_config).resolve()
    out_dir = Path(args.output_dir).resolve()
    frame_buckets = load_frame_buckets(dataset_config)

    pairs, missing_txt = list_pairs(dataset_root)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    print(
        f"[stat] dataset_root={dataset_root} mp4_with_txt={len(pairs)} "
        f"missing_txt={len(missing_txt)} fps={args.fps} frame_buckets={frame_buckets} "
        f"workers={args.workers} output_dir={out_dir}",
        flush=True,
    )

    results = []
    started = time.time()
    tasks = [(str(path), dataset_root, args.fps, frame_buckets) for path in pairs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(stat_one, task) for task in tasks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            result = fut.result()
            results.append(result)
            if idx % args.progress_every == 0 or idx == len(futures):
                ok = sum(1 for r in results if r.get("ok"))
                errors = len(results) - ok
                elapsed = time.time() - started
                print(
                    f"[stat] progress={idx}/{len(futures)} ok={ok} errors={errors} "
                    f"elapsed_sec={elapsed:.1f}",
                    flush=True,
                )

    results.sort(key=lambda r: r["key"])
    summary = write_outputs(out_dir, results, missing_txt, dataset_root, frame_buckets, args.fps)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    sys.exit(main())
