# SimpleOCR: Visual Uncertainty-Aware Reinforcement Learning for Multimodal Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Data](https://img.shields.io/badge/🤗%20Data-HuggingFace-yellow)](https://huggingface.co/datasets/pybbb/simpleocr)
[![GitHub](https://img.shields.io/github/stars/aiming-lab/SimpleOCR)](https://github.com/aiming-lab/SimpleOCR)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of **SimpleOCR**, a novel approach for enhancing multimodal large language models through visual uncertainty-aware reinforcement learning.

## 🔑 Key Features

- **Dual-Branch Training**: Contrasts model behavior on original images vs. text-overlaid images
- **Visual Uncertainty Quantification**: Token-level symmetric KL divergence to measure visual perception confidence
- **Uncertainty-Aware Advantage Estimation**: Incorporates visual uncertainty into GRPO advantage function
- **Built on EasyR1**: Efficient and scalable RL training framework

## 📊 Datasets

All datasets are available on HuggingFace: [`pybbb/simpleocr`](https://huggingface.co/datasets/pybbb/simpleocr)

| Config | Description |
|--------|-------------|
| `train-branch-a` | GRPO Baseline training data (original images) |
| `train-branch-b` | SimpleOCR training data (text-overlaid images) |
| `validation` | Validation dataset |
| `test-ood` | OOD test set (MathVista, MathVision, OCRBench, etc.) |
| `test-chartqa` | ChartQA test set |
| `test-infodocvqa` | InfoDocVQA test set |

```python
from datasets import load_dataset

# Load training data
ds = load_dataset("pybbb/simpleocr", "train-branch-b")

# Load test data
ds_test = load_dataset("pybbb/simpleocr", "test-ood")
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/aiming-lab/SimpleOCR.git
cd SimpleOCR
pip install -e .
```

Or use Docker:
```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

### Training

**GRPO Baseline:**
```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

**SimpleOCR (with visual uncertainty):**
```bash
bash examples/qwen2_5_vl_7b_simpleocr.sh
```

### Inference

```bash
# Start vLLM server
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_PATH \
    --port 8001 \
    --trust-remote-code

# Run inference
python scripts/inference.py \
    --api_base http://localhost:8001/v1 \
    --model_name YOUR_MODEL_NAME \
    --test_data data/test.parquet \
    --output_file results/predictions.jsonl \
    --prompt_template grpo
```

### Evaluation

```bash
python scripts/evaluation.py \
    --results_path results/predictions.jsonl \
    --output_path results/evaluation.jsonl \
    --use_vllm \
    --api_base http://localhost:8002/v1 \
    --evaluator_model Qwen/Qwen2.5-32B-Instruct
```

## 📁 Project Structure

```
SimpleOCR/
├── verl/                    # Core training framework
├── examples/
│   ├── config.yaml          # Training configuration
│   ├── qwen2_5_vl_7b_simpleocr.sh  # SimpleOCR training script
│   └── reward_function/     # Reward functions
├── scripts/
│   ├── data/                # Data processing scripts
│   ├── inference.py         # Inference script
│   ├── evaluation.py        # Evaluation script
│   └── upload_to_hf.py      # HuggingFace upload script
├── docs/
│   ├── QUICKSTART.md        # Quick start guide
│   ├── USAGE_GUIDE.md       # Detailed usage guide
│   └── REPRODUCE.md         # Reproduce experiments
└── METHOD_SECTION.md        # Technical details
```

## 🔧 Configuration

Key configuration options in `examples/config.yaml`:

```yaml
data:
  enable_dual_branch: true   # Enable dual-branch training
  branch: B                  # Use text-overlaid images

worker:
  actor:
    visual_uncertainty_coef_alpha: 1.0  # Enable uncertainty-aware training
    visual_uncertainty_coef_beta: 2.0
    token_entropy_coef_alpha: 0.01
```

## 📖 Documentation

- [Quick Start Guide](docs/QUICKSTART.md)
- [Detailed Usage Guide](docs/USAGE_GUIDE.md)
- [Reproduce Experiments](docs/REPRODUCE.md)
- [Method Details](METHOD_SECTION.md)
- [HuggingFace Datasets](docs/HF_DATASETS.md)

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{simpleocr2025,
  title  = {SimpleOCR: Visual Uncertainty-Aware Reinforcement Learning for Multimodal Reasoning},
  author = {Your Name},
  year   = {2025},
  url    = {https://github.com/aiming-lab/SimpleOCR}
}
```

This project is built on [EasyR1](https://github.com/hiyouga/EasyR1) and [veRL](https://github.com/volcengine/verl). Please also cite:

```bibtex
@misc{zheng2025easyr1,
  title        = {EasyR1: An Efficient, Scalable, Multi-Modality RL Training Framework},
  author       = {Yaowei Zheng, Junting Lu, Shenzhi Wang, Zhangchi Feng, Dongdong Kuang, Yuwen Xiong},
  howpublished = {\url{https://github.com/hiyouga/EasyR1}},
  year         = {2025}
}
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

We thank the authors of EasyR1 and veRL for providing the high-performance RL training framework.
