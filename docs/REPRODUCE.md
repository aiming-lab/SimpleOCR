# Reproducing SimpleOCR Results

This document provides step-by-step instructions to reproduce the results from our paper.

## Prerequisites

1. **Hardware**: 8x NVIDIA A100 80GB (or equivalent)
2. **Software**: Docker or conda environment with dependencies installed
3. **Data**: Download datasets from Hugging Face

## Step 1: Environment Setup

```bash
# Using Docker (recommended)
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all -v /path/to/data:/data hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0

# Or using conda
conda create -n simpleocr python=3.10
conda activate simpleocr
pip install -e .
```

## Step 2: Download Data

```bash
# Download training data from Hugging Face
# Replace with actual dataset URLs
python -c "
from datasets import load_dataset
ds = load_dataset('pybbb/simpleocr-train')
ds.save_to_disk('./data/train')
"
```

## Step 3: Create Overlay Datasets

### Branch A (Standard GRPO)
No modification needed - use original images.

### Branch B (SimpleOCR)
```bash
# Create overlay dataset
python scripts/data/create_overlay_dataset.py \
    --input data/original_train.parquet \
    --output data/overlay_train.parquet
```

## Step 4: Train Models

### GRPO Baseline (Branch A)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/train_a.json \
    data.enable_dual_branch=false \
    worker.actor.visual_uncertainty_coef_alpha=0 \
    trainer.experiment_name=grpo_baseline_7b \
    trainer.n_gpus_per_node=8
```

### SimpleOCR (Branch B with uncertainty)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=./data/train_b.json \
    data.enable_dual_branch=true \
    data.branch=B \
    worker.actor.visual_uncertainty_coef_alpha=1.0 \
    worker.actor.visual_uncertainty_coef_beta=2.0 \
    worker.actor.token_entropy_coef_alpha=0.01 \
    trainer.experiment_name=simpleocr_7b \
    trainer.n_gpus_per_node=8
```

## Step 5: Merge Checkpoints

```bash
# For GRPO baseline
python scripts/model_merger.py \
    --local_dir checkpoints/simpleocr/grpo_baseline_7b/global_step_200/actor

# For SimpleOCR
python scripts/model_merger.py \
    --local_dir checkpoints/simpleocr/simpleocr_7b/global_step_200/actor
```

## Step 6: Inference

### Start vLLM Server

```bash
# For GRPO baseline
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/grpo_baseline_7b_merged \
    --port 8001 \
    --trust-remote-code

# For SimpleOCR (in another terminal)
CUDA_VISIBLE_DEVICES=1 python -m vllm.entrypoints.openai.api_server \
    --model ./checkpoints/simpleocr_7b_merged \
    --port 8002 \
    --trust-remote-code
```

### Run Inference

```bash
# GRPO baseline
python scripts/inference.py \
    --api_base http://localhost:8001/v1 \
    --model_name grpo_baseline_7b \
    --test_data data/test.parquet \
    --output_file results/grpo_baseline_7b.jsonl \
    --max_workers 16 \
    --prompt_template grpo

# SimpleOCR
python scripts/inference.py \
    --api_base http://localhost:8002/v1 \
    --model_name simpleocr_7b \
    --test_data data/test.parquet \
    --output_file results/simpleocr_7b.jsonl \
    --max_workers 16 \
    --prompt_template grpo
```

## Step 7: Evaluation

### Set up Azure OpenAI (for GPT-4 evaluation)

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-2024-08-06"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"
```

### Run Evaluation

```bash
# Evaluate GRPO baseline
python scripts/evaluation.py \
    --results_path results/grpo_baseline_7b.jsonl \
    --output_path results/grpo_baseline_7b_eval.jsonl \
    --use_azure \
    --azure_deployment "$AZURE_OPENAI_DEPLOYMENT" \
    --only_llm_judge \
    --max_workers 128

# Evaluate SimpleOCR
python scripts/evaluation.py \
    --results_path results/simpleocr_7b.jsonl \
    --output_path results/simpleocr_7b_eval.jsonl \
    --use_azure \
    --azure_deployment "$AZURE_OPENAI_DEPLOYMENT" \
    --only_llm_judge \
    --max_workers 128
```

## Step 8: Ablation Studies

### Overlay Ratio Ablation (25%, 50%, 75%)

```bash
# Create datasets with different ratios
for ratio in 0.25 0.50 0.75; do
    python scripts/data/create_mixed_overlay_dataset.py \
        --input data/train.json \
        --output_json data/train_${ratio/./}percent.json \
        --output_images data/images_${ratio/./}percent \
        --overlay_ratio $ratio
done

# Train with each ratio
for ratio in 25 50 75; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
        config=examples/config.yaml \
        data.train_files=./data/train_${ratio}percent.json \
        trainer.experiment_name=simpleocr_${ratio}percent
done
```

## Expected Results

| Model | Geometry3K | MathVista | ChartQA |
|-------|------------|-----------|---------|
| GRPO Baseline | X.X% | X.X% | X.X% |
| SimpleOCR | X.X% | X.X% | X.X% |

## Troubleshooting

### Out of Memory
- Reduce `worker.rollout.gpu_memory_utilization`
- Enable `worker.actor.offload.offload_params: true`
- Reduce `worker.actor.micro_batch_size_per_device_for_update`

### Training Divergence
- Reduce learning rate: `worker.actor.optim.lr: 5.0e-7`
- Reduce uncertainty coefficient: `worker.actor.visual_uncertainty_coef_alpha: 0.5`

### Slow Inference
- Increase `--max_workers` for parallel inference
- Enable tensor parallelism in vLLM
