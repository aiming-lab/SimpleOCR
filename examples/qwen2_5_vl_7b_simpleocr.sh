#!/bin/bash
# SimpleOCR Training Script for Qwen2.5-VL-7B
# This script trains the model with visual uncertainty-aware GRPO

set -x
export PYTHONUNBUFFERED=1

# Model configuration
MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # Replace with your local path if needed

# Logging
LOG_DIR=logs
mkdir -p ${LOG_DIR}
LOG_FILE=${LOG_DIR}/qwen2_5_vl_7b_simpleocr_$(date +"%Y%m%d_%H%M%S").log

# Training with SimpleOCR (dual-branch with uncertainty)
# Key settings for SimpleOCR:
#   - data.enable_dual_branch=true
#   - data.branch=B (uses text-overlaid images)
#   - worker.actor.visual_uncertainty_coef_alpha=1.0 (enables uncertainty-aware training)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    data.enable_dual_branch=true \
    data.branch=B \
    worker.actor.visual_uncertainty_coef_alpha=1.0 \
    worker.actor.visual_uncertainty_coef_beta=2.0 \
    worker.actor.token_entropy_coef_alpha=0.01 \
    trainer.experiment_name=qwen2_5_vl_7b_simpleocr \
    trainer.n_gpus_per_node=8 \
    2>&1 | tee -a ${LOG_FILE}
