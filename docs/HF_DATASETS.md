# HuggingFace Datasets for SimpleOCR

This document describes the datasets uploaded to HuggingFace and their corresponding configurations.

**HuggingFace Repository:** `pybbb/simpleocr`  
**User:** pybbb

---

## Overview: Available Datasets

All datasets are uploaded to a single repository `pybbb/simpleocr` with different config names:

### Training Datasets

| Config Name | Purpose | Description |
|-------------|---------|-------------|
| `train-branch-a` | GRPO Baseline | Original images |
| `train-branch-b` | SimpleOCR | Images with text overlay |
| `validation` | Validation set | For training validation |

### Test Datasets

| Config Name | Purpose | Included Benchmarks |
|-------------|---------|---------------------|
| `test-ood` | OOD Test Set | MathVista, MathVision, MathVerse, OCRBench, HallusionBench, WeMath |
| `test-chartqa` | VQA Test Set | ChartQA |
| `test-infodocvqa` | VQA Test Set | InfoDocVQA (InfographicsVQA) |

**Loading Examples:**

```python
from datasets import load_dataset

# Load training set Branch A (GRPO Baseline)
ds_a = load_dataset("pybbb/simpleocr", "train-branch-a")

# Load training set Branch B (SimpleOCR with overlay)
ds_b = load_dataset("pybbb/simpleocr", "train-branch-b")

# Load OOD test set
ds_test = load_dataset("pybbb/simpleocr", "test-ood")

# Load VQA test sets
ds_chartqa = load_dataset("pybbb/simpleocr", "test-chartqa")
ds_infodocvqa = load_dataset("pybbb/simpleocr", "test-infodocvqa")
```

---

## 1. Training Dataset Branch A (GRPO Baseline)

Training dataset with **original images** (no text overlay).

### Dataset Card

```markdown
---
license: apache-2.0
task_categories:
  - visual-question-answering
  - image-to-text
language:
  - en
tags:
  - math
  - geometry
  - reasoning
  - multimodal
size_categories:
  - 1K<n<10K
---

# SimpleOCR Training Dataset - Branch A (Standard)

This dataset is used for training the GRPO baseline in SimpleOCR. It contains geometry and math problems with **original images** (no text overlay).

## Dataset Description

- **Task**: Visual Math Reasoning
- **Domains**: Geometry3K, K12-Freeform
- **Image Type**: Original problem images

## Dataset Format

```json
{
  "id": "geometry3k_0001",
  "question": "<image>Find the value of x.",
  "answer": "45",
  "image_path": ["images/geometry3k/train/001.png"],
  "dataset": "geometry3k"
}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("pybbb/simpleocr", "train-branch-a")
```

## Related Datasets

- [train-branch-b](https://huggingface.co/datasets/pybbb/simpleocr) - Training with text overlay
- [test-ood](https://huggingface.co/datasets/pybbb/simpleocr) - OOD test set

## Citation

```bibtex
@misc{simpleocr2025,
  title={SimpleOCR: Visual Uncertainty-Aware RL for Multimodal Reasoning},
  year={2025}
}
```
```

---

## 2. Training Dataset Branch B (SimpleOCR with Overlay)

Training dataset with **text-overlaid images**. The question text is rendered directly onto the image.

### Dataset Card

```markdown
---
license: apache-2.0
task_categories:
  - visual-question-answering
  - image-to-text
language:
  - en
tags:
  - math
  - geometry
  - reasoning
  - multimodal
  - ocr
size_categories:
  - 1K<n<10K
---

# SimpleOCR Training Dataset - Branch B (Text Overlay)

This dataset is used for training SimpleOCR with **text-overlaid images**. The question text is rendered directly onto the image.

## Dataset Description

- **Task**: Visual Math Reasoning with OCR
- **Domains**: Geometry3K, K12-Freeform
- **Image Type**: Images with question text overlaid at the bottom

## Key Difference from Branch A

In Branch B, the question text is rendered onto the image itself, and the text prompt is simplified to:
> "Please answer the question shown in the image."

This forces the model to read and understand the visual text, enabling visual uncertainty quantification.

## Dataset Format

```json
{
  "id": "geometry3k_0001",
  "question": "<image>Please answer the question shown in the image.",
  "answer": "45",
  "image_path": ["images/geometry3k_overlay/train/001_overlay.png"],
  "problem": "Find the value of x.",
  "has_overlay": true,
  "dataset": "geometry3k"
}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("pybbb/simpleocr", "train-branch-b")
```

## Citation

```bibtex
@misc{simpleocr2025,
  title={SimpleOCR: Visual Uncertainty-Aware RL for Multimodal Reasoning},
  year={2025}
}
```
```

---

## 3. OOD Test Dataset

Out-of-distribution (OOD) test dataset for evaluating SimpleOCR models.

### Dataset Card

```markdown
---
license: apache-2.0
task_categories:
  - visual-question-answering
  - image-to-text
language:
  - en
tags:
  - math
  - geometry
  - chart
  - ocr
  - reasoning
  - benchmark
size_categories:
  - 1K<n<10K
---

# SimpleOCR OOD Test Dataset

Out-of-distribution (OOD) test dataset for evaluating SimpleOCR models. This dataset includes samples from multiple benchmarks that are **not seen during training**.

## Included Benchmarks

| Benchmark | Domain |
|-----------|--------|
| MathVista | Math reasoning |
| MathVision | Visual math |
| MathVerse | Math problems |
| OCRBench | Text recognition |
| HallusionBench | Hallucination detection |
| WeMath | Math problems |

## Dataset Format

```json
{
  "id": "mathvista_001",
  "dataset": "MathVista",
  "split": "ood_test",
  "question": "What is the sum of the values?",
  "answer": "150",
  "images": [{"bytes": "...", "path": null}],
  "question_type": "free_form",
  "answer_type": "integer",
  "choices": null,
  "precision": 2
}
```

## Usage

```python
from datasets import load_dataset

dataset = load_dataset("pybbb/simpleocr", "test-ood")

# Filter by specific benchmark
mathvista = dataset.filter(lambda x: x['dataset'] == 'MathVista')
```

## Evaluation

Use our evaluation script:

```bash
python scripts/evaluation.py \
    --results_path predictions.jsonl \
    --output_path evaluation.jsonl \
    --use_azure
```

## Citation

```bibtex
@misc{simpleocr2025,
  title={SimpleOCR: Visual Uncertainty-Aware RL for Multimodal Reasoning},
  year={2025}
}
```
```

---

## 4. VQA Test Datasets

### ChartQA Test Set

Chart understanding benchmark.

```python
from datasets import load_dataset
ds_chartqa = load_dataset("pybbb/simpleocr", "test-chartqa")
```

### InfoDocVQA Test Set

Infographics document VQA benchmark.

```python
from datasets import load_dataset
ds_infodocvqa = load_dataset("pybbb/simpleocr", "test-infodocvqa")
```

---

## 5. Upload Script Example

```python
#!/usr/bin/env python3
"""Upload SimpleOCR datasets to HuggingFace"""

import json
from datasets import Dataset, Features, Value, Image, Sequence
from huggingface_hub import HfApi, login
from pathlib import Path
from PIL import Image as PILImage
import io

# Login to HuggingFace
login()

def load_json_dataset(json_path, image_base_dir):
    """Load JSON dataset with images"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    processed = []
    for item in data:
        # Load image
        img_path = item['image_path']
        if isinstance(img_path, list):
            img_path = img_path[0]
        
        full_path = Path(image_base_dir) / img_path
        if full_path.exists():
            img = PILImage.open(full_path)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            item['image'] = img_bytes.getvalue()
        
        processed.append(item)
    
    return Dataset.from_list(processed)

# Upload Branch A
print("Uploading Branch A dataset...")
ds_a = load_json_dataset(
    './data/train_single_a.json',
    './data'
)
ds_a.push_to_hub('pybbb/simpleocr', config_name='train-branch-a')

# Upload Branch B
print("Uploading Branch B dataset...")
ds_b = load_json_dataset(
    './data/train_single_b.json',
    './data'
)
ds_b.push_to_hub('pybbb/simpleocr', config_name='train-branch-b')

# Upload validation set
print("Uploading validation dataset...")
ds_val = load_json_dataset(
    './data/validation.json',
    './data'
)
ds_val.push_to_hub('pybbb/simpleocr', config_name='validation')

print("Done!")
```

---

## 6. Update Training Config

After uploading, update `examples/config.yaml` to point to HuggingFace:

```yaml
data:
  # Load from HuggingFace (recommended)
  train_files: hf://pybbb/simpleocr/train-branch-b
  val_files: hf://pybbb/simpleocr/validation
  
  # Or use local paths
  # train_files: ./data/train.json
  # val_files: ./data/validation.json
```
