#!/bin/bash

set -x
export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct  # replace it with your local file path
LOG_DIR=logs
mkdir -p ${LOG_DIR}

LOG_FILE=${LOG_DIR}/qwen2_5_vl_3b_no_dual_branch_ding_debug_$(date +"%Y%m%d_%H%M%S").log

CUDA_VISIBLE_DEVICES=0,2,4,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_3b_dual_branch_ding_debug_a_branch_only\
    trainer.n_gpus_per_node=4 \
    2>&1 | tee -a ${LOG_FILE}
