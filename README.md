# Multi-Grid Post-Training for Long-Form Multi-Shot Video Generation

**[**[**📄 Paper**](https://arxiv.org/abs/2510.20822)**]**
**[**[**🌐 Project Page**](https://holo-cine.github.io/)**]**
**[**[**🤗 Model Weights**](https://huggingface.co/JiaMao/StoryGrid)**]**

https://github.com/user-attachments/assets/0b91b967-895f-442e-ac35-76bdbbb95fd9

## 🎬 TLDR

* **What it is:** A multi-grid post-training framework for generating long-form multi-shot videos, rather than isolated short clips.
* **Problem It Solves:** It alleviates the trade-off between maintaining continuous motion within shots and generating a growing number of coherent shot transitions.
* **How it works:** It distributes the narrative across a spatial grid of short subvideos, reducing the temporal horizon and number of transitions modeled by each cell, while jointly generating all cells to preserve cross-shot coordination.

## 🔥 News

### ✅ Released
- [x] `16 grid training code`
- [x] `16 grid inference code`
- [x] `64 grid training configuration`
- [x] `64 grid inference configuration`
- [x] `16 grid dataset pipeline`
- [x] `64 grid dataset pipeline`
- [x] `StoryGrid-16 weight` (For 16 grid video generation)
- [x] `StoryGrid-64 weight` (For 64 grid video generation)
- [x] `MGLV dataset pipeline`

### 🗺️ Future Work
- [ ] `support first frame condition`
- [ ] `support storyboard condition`

## 🛠️ Setup

```shell
git clone <repository-url> StoryGrid
cd StoryGrid

# The inference entry point imports the official Wan2.2 implementation.
git clone https://github.com/Wan-Video/Wan2.2.git
```

## 🧰 Environment

```shell
conda create -n StoryGrid python=3.10 -y
conda activate StoryGrid
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

## 📦 Checkpoints

```text
checkpoints/
├── Wan2.2-TI2V-5B/
│   ├── config.json
│   ├── configuration.json
│   ├── diffusion_pytorch_model-00001-of-00003.safetensors
│   ├── diffusion_pytorch_model-00002-of-00003.safetensors
│   ├── diffusion_pytorch_model-00003-of-00003.safetensors
│   ├── diffusion_pytorch_model.safetensors.index.json
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   ├── Wan2.2_VAE.pth
│   └── google/umt5-xxl/
│       ├── special_tokens_map.json
│       ├── spiece.model
│       ├── tokenizer.json
│       └── tokenizer_config.json
└── StoryGrid/
    ├── StoryGrid-16/
    │   ├── adapter_config.json
    │   └── adapter_model.safetensors
    └── StoryGrid-64/
        ├── adapter_config.json
        └── adapter_model.safetensors
```

Training and inference use [Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) as the base model.

```shell
hf download Wan-AI/Wan2.2-TI2V-5B \
  --local-dir checkpoints/Wan2.2-TI2V-5B
```

Use `checkpoints/Wan2.2-TI2V-5B` as `WAN_MODEL_DIR` for both training and inference.

Download the [StoryGrid-16 checkpoint](https://huggingface.co/JiaMao/StoryGrid/tree/main/storygrid-16) for 16-grid inference:

```shell
hf download JiaMao/StoryGrid \
  --include "StoryGrid-16/*" \
  --local-dir checkpoints/StoryGrid
```

For 16 grid video generation, use `checkpoints/StoryGrid/StoryGrid-16` as `ADAPTER_DIR`.

Download the [StoryGrid-64 checkpoint](https://huggingface.co/JiaMao/StoryGrid/tree/main/storygrid-64) for 64-grid inference:

```shell
hf download JiaMao/StoryGrid \
  --include "StoryGrid-64/*" \
  --local-dir checkpoints/StoryGrid
```

For 64 grid video generation, use `checkpoints/StoryGrid/StoryGrid-64` as `ADAPTER_DIR`.

## 🗂️ MGLV Dataset Pipeline

The workflow for building MGLV dataset is available in the [MGLV dataset pipeline documentation](dataset_pipeline/README.md).


## 🚀 Training

Generate the text-prefix embedding once before training:

```shell
python env/generate_wan22_prefix_embedding.py \
  --ckpt_dir checkpoints/Wan2.2-TI2V-5B
```

Generate a separate prefix embedding for 64-grid training:

```shell
python env/generate_wan22_prefix_embedding.py \
  --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
  --text "<grid 64>" \
  --output env/prefix_embeddings/wan22_64grid_prefix.pt
```

Start 16-grid LoRA training:

```shell
WAN_MODEL_DIR=checkpoints/Wan2.2-TI2V-5B \
DATASET_DIR=/path/to/cached-16grid-dataset \
NUM_GPUS=8 \
bash scripts/train.sh
```

Training outputs are saved in `outputs/16grid_lora`.

The same training code supports 64-grid LoRA training. Launch training with the 64-grid configuration on 8x8 cached dataset:

```shell
WAN_MODEL_DIR=checkpoints/Wan2.2-TI2V-5B \
DATASET_DIR=/path/to/cached-64grid-dataset \
CONFIG="${PWD}/env/configs/train_64grid_lora.toml" \
PREFIX_EMBEDDING_PATH=env/prefix_embeddings/wan22_64grid_prefix.pt \
OUTPUT_DIR=outputs/64grid_lora \
NUM_GPUS=8 \
bash scripts/train.sh
```

## 🔮 Inference

```shell
WAN_REPO=Wan2.2 \
WAN_MODEL_DIR=checkpoints/Wan2.2-TI2V-5B \
bash scripts/infer.sh \
  checkpoints/StoryGrid/StoryGrid-16 \
  env/inference_prompts/16grid/3dcgi_boy_robot_cat_fair_short.txt \
  outputs/StoryGrid-16
```

Generated videos are saved in `outputs/StoryGrid-16`.

For 64-grid inference, use the StoryGrid-64 LoRA and explicitly select the 8x8 visual-slot layout:

```shell
WAN_REPO=Wan2.2 \
WAN_MODEL_DIR=checkpoints/Wan2.2-TI2V-5B \
bash scripts/infer.sh \
  checkpoints/StoryGrid/StoryGrid-64 \
  env/inference_prompts/64grid/cinematic_neural_courier_hospital_airport.txt \
  outputs/StoryGrid-64 \
  --visual_slot_count 64 \
  --visual_slot_rows 8 \
  --visual_slot_cols 8 \
  --prompt_prefix_text "<grid 64>"
```

## 🙏 Acknowledgements

Deeply appreciate these wonderful open source projects: [Wan2.2](https://github.com/Wan-Video/Wan2.2), [HoloCine](https://holo-cine.github.io/), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), [diffusion-pipe](https://github.com/tdrussell/diffusion-pipe), [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [PyTorch](https://github.com/pytorch/pytorch), [Transformers](https://github.com/huggingface/transformers), [Diffusers](https://github.com/huggingface/diffusers), and [PEFT](https://github.com/huggingface/peft). 

## 🔖 Citation 

If you find this repository useful, please consider giving a star ⭐ and citation 🙈:

```
@inproceedings{maostory,
  title={Story-Iter: A Training-free Iterative Paradigm for Long Story Visualization},
  author={Mao, Jiawei and Huang, Xiaoke and Xie, Yunfei and Chang, Yuanqi and Hui, Mude and Xu, Bingjie and Zheng, Zeyu and Wang, Zirui and Xie, Cihang and Zhou, Yuyin},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
