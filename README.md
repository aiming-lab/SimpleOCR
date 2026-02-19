# SimpleOCR: Visual Uncertainty-Aware Reinforcement Learning for Multimodal Reasoning

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![Data](https://img.shields.io/badge/🤗%20Data-HuggingFace-yellow)](https://huggingface.co/datasets/simpleocr/simpleocr)
[![GitHub](https://img.shields.io/github/stars/aiming-lab/SimpleOCR)](https://github.com/aiming-lab/SimpleOCR)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

This repository contains the official implementation of **SimpleOCR**, a novel approach for enhancing multimodal large language models through visual uncertainty-aware reinforcement learning.

## 📊 Datasets

All datasets are available on HuggingFace: [`simpleocr/simpleocr`](https://huggingface.co/datasets/simpleocr/simpleocr)

| Config | Description |
|--------|-------------|
| `train-branch-a` | GRPO Baseline training data (original images) |
| `train-branch-b` | SimpleOCR training data (text-overlaid images) |
| `validation` | Validation set |
| `test-ood` | OOD test set (MathVista, MathVision, OCRBench, etc.) |
| `test-chartqa` | ChartQA test set |
| `test-infodocvqa` | InfoDocVQA test set |

```python
from datasets import load_dataset

# Load Branch-A training data (original images)
ds_a = load_dataset("simpleocr/simpleocr", "train-branch-a", split="train")

# Load Branch-B training data (text-overlaid images)
ds_b = load_dataset("simpleocr/simpleocr", "train-branch-b", split="train")

# Load test sets
ds_test = load_dataset("simpleocr/simpleocr", "test-ood", split="train")
```

## 🚀 Quick Start

### Installation

SimpleOCR is built on top of [EasyR1](https://github.com/hiyouga/EasyR1). We recommend setting up the EasyR1 environment first, then installing SimpleOCR on top.

**Option 1: Docker (recommended)**

Use the pre-built EasyR1 Docker image, which includes PyTorch, flash-attn, and vLLM:

```bash
docker pull hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
docker run -it --ipc=host --gpus=all hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0
```

Then inside the container:

```bash
git clone https://github.com/aiming-lab/SimpleOCR.git
cd SimpleOCR
pip install -e .
```

**Option 2: pip**

Follow the [EasyR1 installation guide](https://github.com/hiyouga/EasyR1) to set up the base environment (Python 3.10+, PyTorch, flash-attn, vLLM), then:

```bash
git clone https://github.com/aiming-lab/SimpleOCR.git
cd SimpleOCR
pip install -e .
```

### Prepare Training Data

SimpleOCR uses a **dual-branch** design:
- **Branch A** — original images, used as the GRPO baseline
- **Branch B** — question text overlaid onto images, used for SimpleOCR uncertainty-aware training

Choose one of the two paths below depending on whether you use our dataset or your own.

---

#### Path 1: Use our pre-built dataset (recommended)

**Step 1 — Download from HuggingFace and save as Parquet**

```python
from datasets import load_dataset

ds_b   = load_dataset("simpleocr/simpleocr", "train-branch-b", split="train")
ds_val = load_dataset("simpleocr/simpleocr", "validation",     split="train")

ds_b.to_parquet("data/train_branch_b.parquet")
ds_val.to_parquet("data/validation.parquet")
```

**Step 2 — Convert Parquet to JSON + image directory (required by the training framework)**

```bash
python scripts/convert_parquet_to_json.py \
    --input   data/train_branch_b.parquet \
    --output  data/train_branch_b.json \
    --image_dir data/images/train_branch_b

python scripts/convert_parquet_to_json.py \
    --input   data/validation.parquet \
    --output  data/validation.json \
    --image_dir data/images/validation
```

**Step 3 — Update `config.yaml`**

```yaml
data:
  train_files: ./data/train_branch_b.json
  val_files:   ./data/validation.json
  image_dir:   ./data/images
```

---

#### Path 2: Use your own dataset

Your dataset must be a JSON file where each item contains at minimum:
`id`, `question`, `answer`, `image_path`

**Step 1 — Convert JSON to Parquet (Branch-A)**

```bash
python scripts/convert_to_parquet_hf.py \
    --json /path/to/your_data.json \
    --output data/train_branch_a.parquet

# If image paths in the JSON are relative, add --image_base_dir:
python scripts/convert_to_parquet_hf.py \
    --json /path/to/your_data.json \
    --output data/train_branch_a.parquet \
    --image_base_dir /path/to/images/
```

**Step 2 — Generate Branch-B by overlaying question text onto images**

```bash
python scripts/create_overlay_dataset.py \
    --input data/train_branch_a.parquet \
    --output data/train_branch_b.parquet
```

**Step 3 — Convert Parquet to JSON + image directory (required by the training framework)**

```bash
python scripts/convert_parquet_to_json.py \
    --input   data/train_branch_b.parquet \
    --output  data/train_branch_b.json \
    --image_dir data/images/train_branch_b
```

**Step 4 — Update `config.yaml`**

```yaml
data:
  train_files: ./data/train_branch_b.json
  val_files:   ./data/validation.json
  image_dir:   ./data/images
```

---

### Training

**GRPO Baseline (Branch-A only, no overlay):**
```bash
bash examples/qwen2_5_vl_7b_geo3k_grpo.sh
```

**SimpleOCR (Branch-B, overlay, with visual question):**
```bash
bash examples/qwen2_5_vl_7b_simpleocr.sh
```

### Inference

**Step 1 — Download test sets from HuggingFace**

```python
from datasets import load_dataset
import os

os.makedirs("data/test", exist_ok=True)

for config in ["test-ood", "test-chartqa", "test-infodocvqa"]:
    ds = load_dataset("simpleocr/simpleocr", config, split="train")
    ds.to_parquet(f"data/test/{config}.parquet")
    print(f"Saved {config} ({len(ds)} samples)")
```

**Step 2 — Start the vLLM server**

```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model YOUR_MODEL_PATH \
    --port 15270
```

**Step 3 — Run inference on each test set**

```bash
for TEST_SET in test-ood test-chartqa test-infodocvqa; do
    python scripts/inference.py \
        --api_base http://localhost:15270/v1 \
        --model_name YOUR_MODEL_PATH \
        --test_data data/test/${TEST_SET}.parquet \
        --output_file results/${TEST_SET}_predictions.jsonl \
        --max_workers 16 \
        --prompt_template grpo
done
```

### Evaluation

Evaluation uses an LLM judge (GPT-4o). Set your credentials as environment variables, then run evaluation on each test set.

**Using Azure OpenAI:**

```bash
export AZURE_OPENAI_ENDPOINT="https://YOUR_ENDPOINT.azure-api.net"
export AZURE_OPENAI_API_KEY="YOUR_AZURE_API_KEY"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-2024-08-06"
export AZURE_OPENAI_API_VERSION="2024-08-01-preview"

for TEST_SET in test-ood test-chartqa test-infodocvqa; do
    python scripts/evaluation.py \
        --results_path results/${TEST_SET}_predictions.jsonl \
        --output_path  results/${TEST_SET}_eval.jsonl \
        --use_azure \
        --azure_deployment "$AZURE_OPENAI_DEPLOYMENT" \
        --only_llm_judge \
        --max_workers 256
done
```

**Using standard OpenAI:**

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"

for TEST_SET in test-ood test-chartqa test-infodocvqa; do
    python scripts/evaluation.py \
        --results_path results/${TEST_SET}_predictions.jsonl \
        --output_path  results/${TEST_SET}_eval.jsonl \
        --only_llm_judge \
        --max_workers 256
done
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
│   ├── convert_to_parquet_hf.py   # Convert JSON dataset to Parquet (for HF upload)
│   ├── convert_parquet_to_json.py # Convert Parquet (embedded images) to JSON + image dir
│   ├── create_overlay_dataset.py  # Overlay question text onto images (Branch-B)
│   ├── inference.py               # Inference script
│   ├── evaluation.py              # Evaluation script
│   └── upload_to_hf.py            # HuggingFace upload script
├── docs/
│   ├── QUICKSTART.md        # Quick start guide
│   ├── USAGE_GUIDE.md       # Detailed usage guide
│   └── REPRODUCE.md         # Reproduce experiments
└── METHOD_SECTION.md        # Technical details
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
