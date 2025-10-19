#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=qwen2_5_vl_7b_all_grpo_new \
    trainer.n_gpus_per_node=4
