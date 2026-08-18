#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / 'diffusion-pipe'))
torch.cuda.current_device = lambda: 0

from models.wan.t5 import T5EncoderModel  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--text', default='<grid 16>')
    parser.add_argument(
        '--output',
        default=str(REPO_ROOT / 'env' / 'prefix_embeddings' / 'wan22_16grid_prefix.pt'),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_dir = Path(args.ckpt_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device='cpu',
        checkpoint_path=str(ckpt_dir / 'models_t5_umt5-xxl-enc-bf16.pth'),
        tokenizer_path=str(ckpt_dir / 'google/umt5-xxl'),
        shard_fn=None,
    )
    ids, mask = model.tokenizer([args.text], return_mask=True, add_special_tokens=True)
    ids = ids.to('cpu')
    mask = mask.to('cpu')
    seq_len = int(mask.gt(0).sum(dim=1).item())
    with torch.no_grad():
        text_embeddings = model.model(ids, mask)[0][:seq_len].to(torch.float32).cpu()

    torch.save(
        {
            'text': args.text,
            'text_embeddings': text_embeddings,
            'seq_len': seq_len,
        },
        output_path,
    )
    print(output_path)


if __name__ == '__main__':
    main()
