#!/usr/bin/env python3
"""
Upload SimpleOCR datasets to HuggingFace
Organization: simpleocr
Repository: simpleocr/simpleocr

Usage:
    # Set your token as environment variable (secure)
    export HF_TOKEN="your_token_here"
    
    # Or login via CLI
    huggingface-cli login
    
    # Then run this script
    python scripts/upload_to_hf.py --upload_train
    python scripts/upload_to_hf.py --upload_test
    python scripts/upload_to_hf.py --upload_all
"""

import os
import sys
import json
import argparse
from pathlib import Path
from tqdm import tqdm

# Check dependencies
try:
    from datasets import Dataset, load_dataset
    from huggingface_hub import HfApi, login
    from PIL import Image
    import io
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install: pip install datasets huggingface-hub pillow")
    sys.exit(1)

# HuggingFace configuration
HF_ORG = "simpleocr"
HF_REPO_PREFIX = f"{HF_ORG}/simpleocr"

# Dataset paths (adjust if needed)
DATASET_PATHS = {
    # Training datasets
    "train_branch_a": {
        "json": "/home/yibop/grpo_dataset/train_single_a.json",
        "base_dir": "/home/yibop/grpo_dataset",
        "config_name": "train-branch-a",
        "description": "GRPO Baseline training data (original images)"
    },
    "train_branch_b": {
        "json": "/home/yibop/grpo_dataset/train_single_b.json",
        "base_dir": "/home/yibop/grpo_dataset",
        "config_name": "train-branch-b",
        "description": "SimpleOCR training data (text-overlaid images)"
    },
    "validation": {
        "json": "/home/yibop/grpo_dataset/validation.json",
        "base_dir": "/home/yibop/grpo_dataset",
        "config_name": "validation",
        "description": "Validation dataset"
    },
    # Test datasets - OOD
    "test_ood": {
        "parquet": "/home/yibop/processed_datasets_no_overlay_test/merged/ood_test.parquet",
        "config_name": "test-ood",
        "description": "OOD test set (MathVista, MathVision, OCRBench, HallusionBench, etc.)"
    },
    # Test datasets - VQA
    "test_chartqa": {
        "parquet": "/home/yibop/processed_datasets_no_overlay_test/chartqa_test.parquet",
        "config_name": "test-chartqa",
        "description": "ChartQA test set"
    },
    "test_infodocvqa": {
        "json": "/home/yibop/backup/data/infodocvqa/infographicsvqa_qas/infographicsVQA_test_v1.0.json",
        "img_root": "/home/yibop/backup/data/infodocvqa/infographicsvqa_images",
        "config_name": "test-infodocvqa",
        "description": "InfoDocVQA (InfographicsVQA) test set"
    }
}


def authenticate():
    """Authenticate with HuggingFace"""
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print(f"✓ Authenticated with HF_TOKEN environment variable")
    else:
        # Try to use cached credentials
        try:
            api = HfApi()
            user = api.whoami()
            print(f"✓ Authenticated as: {user['name']}")
        except Exception:
            print("✗ Not authenticated. Please run: huggingface-cli login")
            print("  Or set HF_TOKEN environment variable")
            sys.exit(1)


def load_json_dataset_with_images(json_path, base_dir):
    """Load JSON dataset and embed images"""
    print(f"Loading: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed = []
    skipped = 0
    
    for item in tqdm(data, desc="Processing samples"):
        try:
            img_path = item.get('image_path', [])
            if isinstance(img_path, list):
                img_path = img_path[0] if img_path else None
            
            if img_path:
                full_path = Path(base_dir) / img_path
                if full_path.exists():
                    img = Image.open(full_path).convert('RGB')
                    img_bytes = io.BytesIO()
                    img.save(img_bytes, format='PNG')
                    item['image_bytes'] = img_bytes.getvalue()
                else:
                    skipped += 1
                    continue
            
            processed.append(item)
        except Exception as e:
            print(f"Warning: Failed to process item: {e}")
            skipped += 1
            continue
    
    print(f"Processed: {len(processed)}, Skipped: {skipped}")
    return Dataset.from_list(processed)


def upload_json_dataset(name, config):
    """Upload a JSON-based dataset"""
    print(f"\n{'='*60}")
    print(f"Uploading: {name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    if not Path(config['json']).exists():
        print(f"✗ File not found: {config['json']}")
        return False
    
    ds = load_json_dataset_with_images(config['json'], config['base_dir'])
    
    repo_id = HF_REPO_PREFIX
    config_name = config['config_name']
    
    print(f"Pushing to: {repo_id} (config: {config_name})")
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        private=False
    )
    print(f"✓ Uploaded successfully!")
    return True


def upload_parquet_dataset(name, config):
    """Upload a Parquet-based dataset"""
    print(f"\n{'='*60}")
    print(f"Uploading: {name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    if not Path(config['parquet']).exists():
        print(f"✗ File not found: {config['parquet']}")
        return False
    
    ds = load_dataset('parquet', data_files=config['parquet'], split='train')
    
    repo_id = HF_REPO_PREFIX
    config_name = config['config_name']
    
    print(f"Pushing to: {repo_id} (config: {config_name})")
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        private=False
    )
    print(f"✓ Uploaded successfully!")
    return True


def upload_infodocvqa_dataset(name, config):
    """Upload InfoDocVQA dataset (JSON + separate image directory)"""
    print(f"\n{'='*60}")
    print(f"Uploading: {name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    json_path = Path(config['json'])
    img_root = Path(config['img_root'])
    
    if not json_path.exists():
        print(f"✗ JSON not found: {json_path}")
        return False
    
    if not img_root.exists():
        print(f"✗ Image directory not found: {img_root}")
        return False
    
    print(f"Loading: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # InfoDocVQA format: {"data": [{"questionId": ..., "question": ..., "image_local_name": ..., "answers": [...]}]}
    items = data.get('data', data)  # Handle both formats
    
    processed = []
    skipped = 0
    
    for item in tqdm(items, desc="Processing InfoDocVQA"):
        try:
            # Get image path
            img_name = item.get('image_local_name', item.get('image', ''))
            if not img_name:
                skipped += 1
                continue
            
            img_path = img_root / img_name
            if not img_path.exists():
                skipped += 1
                continue
            
            # Load and embed image
            img = Image.open(img_path).convert('RGB')
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG')
            
            # Normalize format
            answers = item.get('answers', [])
            if isinstance(answers, list) and len(answers) > 0:
                answer = answers[0] if isinstance(answers[0], str) else str(answers[0])
            else:
                answer = str(answers) if answers else ""
            
            processed.append({
                'id': str(item.get('questionId', item.get('id', len(processed)))),
                'question': item.get('question', ''),
                'answer': answer,
                'answers': answers,
                'image_bytes': img_bytes.getvalue(),
                'image_name': img_name,
                'dataset': 'InfoDocVQA'
            })
        except Exception as e:
            print(f"Warning: Failed to process item: {e}")
            skipped += 1
            continue
    
    print(f"Processed: {len(processed)}, Skipped: {skipped}")
    
    ds = Dataset.from_list(processed)
    
    repo_id = HF_REPO_PREFIX
    config_name = config['config_name']
    
    print(f"Pushing to: {repo_id} (config: {config_name})")
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        private=False
    )
    print(f"✓ Uploaded successfully!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Upload SimpleOCR datasets to HuggingFace")
    parser.add_argument("--upload_train", action="store_true", help="Upload training datasets (branch A, B, validation)")
    parser.add_argument("--upload_test", action="store_true", help="Upload OOD test dataset")
    parser.add_argument("--upload_vqa", action="store_true", help="Upload VQA test datasets (ChartQA, InfoDocVQA)")
    parser.add_argument("--upload_all", action="store_true", help="Upload all datasets")
    parser.add_argument("--dry_run", action="store_true", help="Show what would be uploaded without actually uploading")
    args = parser.parse_args()
    
    if not (args.upload_train or args.upload_test or args.upload_vqa or args.upload_all):
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/upload_to_hf.py --upload_all      # Upload everything")
        print("  python scripts/upload_to_hf.py --upload_train    # Only training data")
        print("  python scripts/upload_to_hf.py --upload_test     # Only OOD test set")
        print("  python scripts/upload_to_hf.py --upload_vqa      # Only VQA test sets")
        return
    
    print(f"\n{'='*60}")
    print(f"SimpleOCR Dataset Uploader")
    print(f"Target: {HF_REPO_PREFIX}")
    print(f"{'='*60}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would upload:")
        for name, config in DATASET_PATHS.items():
            if 'parquet' in config:
                print(f"  - {name}: {config['parquet']}")
            elif 'img_root' in config:
                print(f"  - {name}: {config['json']} + {config['img_root']}")
            else:
                print(f"  - {name}: {config['json']}")
        return
    
    # Authenticate
    authenticate()
    
    # Upload training datasets
    if args.upload_train or args.upload_all:
        print("\n>>> Uploading Training Datasets <<<")
        upload_json_dataset("train_branch_a", DATASET_PATHS["train_branch_a"])
        upload_json_dataset("train_branch_b", DATASET_PATHS["train_branch_b"])
        upload_json_dataset("validation", DATASET_PATHS["validation"])
    
    # Upload OOD test dataset
    if args.upload_test or args.upload_all:
        print("\n>>> Uploading OOD Test Dataset <<<")
        upload_parquet_dataset("test_ood", DATASET_PATHS["test_ood"])
    
    # Upload VQA test datasets
    if args.upload_vqa or args.upload_all:
        print("\n>>> Uploading VQA Test Datasets <<<")
        upload_parquet_dataset("test_chartqa", DATASET_PATHS["test_chartqa"])
        upload_infodocvqa_dataset("test_infodocvqa", DATASET_PATHS["test_infodocvqa"])
    
    print(f"\n{'='*60}")
    print("Upload complete!")
    print(f"View at: https://huggingface.co/datasets/{HF_REPO_PREFIX}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
