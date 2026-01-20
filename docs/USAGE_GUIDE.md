# SimpleOCR Usage Guide

This document provides complete instructions for using SimpleOCR.

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Dataset Preparation](#dataset-preparation)
3. [Training](#training)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [Ablation Studies](#ablation-studies)

---

## Environment Setup

### Install Dependencies

```bash
# Create conda environment
conda create -n simpleocr python=3.10
conda activate simpleocr

# Install the package
pip install -e .

# Or install dependencies directly
pip install -r requirements.txt
```

### Environment Variables

```bash
# Copy environment template
cp env.example .env

# Edit .env file with your credentials:
# - AZURE_OPENAI_* for evaluation
# - HF_TOKEN for Hugging Face
# - WANDB_API_KEY for experiment tracking
```

---

## Dataset Preparation

### Dataset Format

Training data should be in JSON format:

```json
[
  {
    "id": "sample_001",
    "question": "<image>What is the value of x in this triangle?",
    "answer": "45",
    "image_path": ["images/geometry/001.png"],
    "dataset": "geometry3k"
  }
]
```

### Create Overlay Dataset

**Full overlay (100% samples with text on images):**

```bash
python scripts/data/create_overlay_dataset.py \
    --input data/original.parquet \
    --output data/overlay.parquet
```

**Mixed overlay (configurable ratio):**

```bash
# 50% overlay + 50% original
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_50percent.json \
    --output_images data/images_50percent \
    --overlay_ratio 0.5 \
    --seed 42
```

---

## Training

### Configuration

Edit `examples/config.yaml`:

```yaml
data:
  train_files: ./data/train.json       # Your training data
  val_files: ./data/validation.json    # Your validation data
  image_dir: ./data/images             # Image directory

  # SimpleOCR settings
  enable_dual_branch: true  # Enable dual-branch training
  branch: B                 # Use text-overlaid images

worker:
  actor:
    # Visual uncertainty coefficients
    visual_uncertainty_coef_alpha: 1.0  # >0 enables uncertainty-aware training
    visual_uncertainty_coef_beta: 2.0
    token_entropy_coef_alpha: 0.01
    token_entropy_coef_beta: 2.0
```

### Run Training

**GRPO Baseline:**

```bash
# Standard GRPO without SimpleOCR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.enable_dual_branch=false \
    trainer.experiment_name=grpo_baseline
```

**SimpleOCR:**

```bash
# SimpleOCR with visual uncertainty
bash examples/qwen2_5_vl_7b_simpleocr.sh

# Or manually:
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.enable_dual_branch=true \
    data.branch=B \
    worker.actor.visual_uncertainty_coef_alpha=1.0 \
    trainer.experiment_name=simpleocr
```

### Merge Checkpoint

```bash
python scripts/model_merger.py \
    --local_dir checkpoints/simpleocr/exp_name/global_step_X/actor
```

---

## Inference

### Step 1: Start vLLM Server

```bash
conda activate simpleocr

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_PATH \
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

**Parameters:**
- `--prompt_template grpo`: Uses `<think>...</think>` and `\boxed{}` format
- `--prompt_template simple`: Direct answer without reasoning tags
- `--max_workers`: Number of concurrent inference threads

---

## Evaluation

### Using Azure OpenAI (GPT-4)

```bash
# Set environment variables
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-2024-08-06"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"

# Run evaluation
python scripts/evaluation.py \
    --results_path results/predictions.jsonl \
    --output_path results/evaluation.jsonl \
    --use_azure \
    --azure_deployment "$AZURE_OPENAI_DEPLOYMENT" \
    --only_llm_judge \
    --max_workers 128
```

### Using Open-Source LLM (via vLLM)

```bash
# Start evaluator model server (in another terminal)
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

### Evaluation Modes

- **Hybrid (default)**: Math-Verify + LLM fallback
- **LLM Only**: `--only_llm_judge` - Always use LLM for judging

---

## Ablation Studies

### Overlay Ratio Ablation

```bash
# 25% overlay
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_25percent.json \
    --output_images data/images_25percent \
    --overlay_ratio 0.25

# 50% overlay
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_50percent.json \
    --output_images data/images_50percent \
    --overlay_ratio 0.50

# 75% overlay
python scripts/data/create_mixed_overlay_dataset.py \
    --input data/train.json \
    --output_json data/train_75percent.json \
    --output_images data/images_75percent \
    --overlay_ratio 0.75
```

### Train with Different Ratios

```bash
# Modify config.yaml to point to the new dataset
# Then run training
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/train_25percent.json \
    trainer.experiment_name=simpleocr_25percent
```

---

## Model Outputs

### Inference Output Format

```json
{
  "id": "sample_001",
  "dataset": "geometry3k",
  "question": "What is the value of x?",
  "answer": "45",
  "model_response": "<think>Looking at the triangle...</think>The answer is \\boxed{45}",
  "model_answer": ["<think>Looking at the triangle...</think>The answer is \\boxed{45}"]
}
```

### Evaluation Output Format

```json
{
  "id": "sample_001",
  "score": 1.0,
  "evaluation": {
    "score": 1.0,
    "method": "math_verify",
    "extraction": "45",
    "math_verify_passed": true
  }
}
```

---

## Tips and Best Practices

1. **Start with smaller models** (3B) for debugging before scaling to 7B/32B
2. **Monitor GPU memory**: Reduce `gpu_memory_utilization` if OOM
3. **Use checkpointing**: Enable `find_last_checkpoint: true` to resume training
4. **Parallel evaluation**: Use high `--max_workers` for faster evaluation
5. **Save intermediate results**: Inference script auto-saves progress on interruption
