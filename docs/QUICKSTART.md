# SimpleOCR Quick Start Guide

This guide helps you get started with SimpleOCR - Visual Uncertainty-Aware Reinforcement Learning for Multimodal Reasoning.

## Overview

SimpleOCR is a method that enhances multimodal LLMs through:
1. **Dual-Branch Training**: Contrasts model behavior on original vs. text-overlaid images
2. **Visual Uncertainty Quantification**: Token-level symmetric KL divergence to measure visual perception confidence
3. **Uncertainty-Aware Advantage Estimation**: Incorporates visual uncertainty into GRPO advantage function

## Installation

### Using Docker (Recommended)

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

### From Source

```bash
git clone https://github.com/pybbb/SimpleOCR.git
cd SimpleOCR
pip install -e .
```

### Environment Setup

```bash
# Copy environment template
cp env.example .env
# Edit .env with your API keys (for evaluation)
```

## Data Preparation

### Option 1: Download Pre-built Datasets

```bash
# Download from Hugging Face
python -c "from datasets import load_dataset; load_dataset('pybbb/simpleocr-geometry3k')"
```

### Option 2: Create Your Own Overlay Dataset

```bash
# Create 100% overlay dataset (all questions overlaid on images)
python scripts/data/create_overlay_dataset.py \
    --input data/original_test.parquet \
    --output data/overlay_test.parquet

# Create mixed dataset (50% overlay + 50% original)
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_mixed.json \
    --output_images data/images_mixed \
    --overlay_ratio 0.5
```

## Training

### Step 1: Configure Training

```bash
# Copy example config
cp examples/config_example.yaml examples/config.yaml

# Edit config.yaml to set your data paths:
# - data.train_files: path to training JSON
# - data.val_files: path to validation JSON  
# - data.image_dir: path to image directory
```

### Step 2: Run Training

**GRPO Baseline (without SimpleOCR):**
```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

**SimpleOCR (with visual uncertainty):**
```bash
bash examples/qwen2_5_vl_7b_simpleocr.sh
```

Key SimpleOCR configuration options:
- `data.enable_dual_branch=true`: Enable dual-branch training
- `data.branch=B`: Use text-overlaid images  
- `worker.actor.visual_uncertainty_coef_alpha=1.0`: Enable uncertainty-aware training

### Step 3: Merge Checkpoint

```bash
python scripts/model_merger.py --local_dir checkpoints/simpleocr/exp_name/global_step_X/actor
```

## Inference

### Step 1: Start vLLM Server

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_PATH_OR_HF_ID \
    --port 8001 \
    --trust-remote-code
```

### Step 2: Run Inference

```bash
python scripts/inference.py \
    --api_base http://localhost:8001/v1 \
    --model_name YOUR_MODEL_NAME \
    --test_data data/test.parquet \
    --output_file results/predictions.jsonl \
    --max_workers 16 \
    --prompt_template grpo
```

## Evaluation

### Option 1: Using Azure OpenAI (GPT-4)

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-2024-08-06"

python scripts/evaluation.py \
    --results_path results/predictions.jsonl \
    --output_path results/evaluation.jsonl \
    --use_azure \
    --azure_deployment "$AZURE_OPENAI_DEPLOYMENT" \
    --only_llm_judge \
    --max_workers 128
```

### Option 2: Using Open-Source LLM via vLLM

```bash
# Start evaluator model server
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-32B-Instruct \
    --port 8002 \
    --trust-remote-code

# Run evaluation
python scripts/evaluation.py \
    --results_path results/predictions.jsonl \
    --output_path results/evaluation.jsonl \
    --use_vllm \
    --api_base http://localhost:8002/v1 \
    --evaluator_model Qwen/Qwen2.5-32B-Instruct
```

## Ablation Studies

### Different Overlay Ratios

```bash
# Create 25% overlay dataset
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_25percent.json \
    --output_images data/images_25percent \
    --overlay_ratio 0.25

# Create 75% overlay dataset
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_75percent.json \
    --output_images data/images_75percent \
    --overlay_ratio 0.75
```

## Project Structure

```
SimpleOCR/
├── verl/                       # Core training framework
│   ├── trainer/               # Training logic
│   ├── workers/               # Actor, rollout, reward workers
│   ├── utils/                 # Utilities including image_text_overlay
│   └── models/                # Model utilities
├── examples/
│   ├── config.yaml            # Training configuration
│   ├── config_example.yaml    # Example configuration template
│   ├── qwen2_5_vl_7b_simpleocr.sh  # SimpleOCR training script
│   ├── qwen2_5_vl_7b_geo3k_grpo.sh # GRPO baseline script
│   ├── format_prompt/         # Prompt templates
│   └── reward_function/       # Reward functions
├── scripts/
│   ├── data/                  # Data processing scripts
│   │   ├── create_overlay_dataset.py
│   │   └── create_mixed_overlay_dataset.py
│   ├── inference.py           # Inference script
│   ├── evaluation.py          # Evaluation script
│   └── model_merger.py        # Checkpoint merger
├── docs/
│   ├── QUICKSTART.md          # This file
│   └── METHOD_SECTION.md      # Technical details
└── env.example                # Environment variables template
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size
worker.actor.micro_batch_size_per_device_for_update: 1

# Enable offloading
worker.actor.offload.offload_params: true

# Reduce GPU memory for vLLM
worker.rollout.gpu_memory_utilization: 0.4
```

### Image Token Mismatch
```bash
# Increase prompt length or reduce image size
data.max_prompt_length: 4096
data.max_pixels: 1048576
```

## Citation

If you use this code, please cite:

```bibtex
@misc{simpleocr2025,
  title  = {SimpleOCR: Visual Uncertainty-Aware Reinforcement Learning for Multimodal Reasoning},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/pybbb/SimpleOCR}
}
```

Also cite the EasyR1/veRL frameworks:

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```
