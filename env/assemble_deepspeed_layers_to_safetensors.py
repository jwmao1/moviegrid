#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors.torch import save_file


def assemble_state_dict(checkpoint_dir: Path) -> OrderedDict[str, torch.Tensor]:
    state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()

    for layer_idx in range(32):
        layer_path = checkpoint_dir / f"layer_{layer_idx:02d}-model_states.pt"
        layer_state = torch.load(layer_path, map_location="cpu")

        if layer_idx == 0:
            for key, value in layer_state.items():
                state_dict[key] = value.contiguous()
        elif layer_idx == 31:
            for key, value in layer_state.items():
                state_dict[key] = value.contiguous()
        else:
            block_idx = layer_idx - 1
            for key, value in layer_state.items():
                if not key.startswith("block."):
                    raise RuntimeError(f"Unexpected key in {layer_path}: {key}")
                state_dict[f"blocks.{block_idx}.{key[len('block.'):]}"] = value.contiguous()

    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_dict = assemble_state_dict(checkpoint_dir)
    save_file(state_dict, output_path, metadata={"format": "pt", "source": str(checkpoint_dir)})
    print(f"saved {len(state_dict)} tensors to {output_path}")


if __name__ == "__main__":
    main()
