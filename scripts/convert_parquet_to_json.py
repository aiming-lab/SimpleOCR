#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a HuggingFace Parquet dataset (with embedded images) to JSON + image directory.
This produces the format expected by the training framework (config.yaml: train_files / image_key).

Output layout:
    <image_dir>/
        <id>_0.png
        <id>_1.png
        ...
    <output_json>
"""

import json
import io
import argparse
import logging
from pathlib import Path

import pandas as pd
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def convert_parquet_to_json(parquet_path: str, output_json: str, image_dir: str):
    """
    Args:
        parquet_path: Path to input Parquet file (images embedded as bytes).
        output_json:  Path to output JSON file.
        image_dir:    Directory where extracted image files will be saved.
    """
    image_dir_path = Path(image_dir)
    image_dir_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logger.info(f"  {len(df)} samples found")

    records = []
    for idx, row in df.iterrows():
        record = {k: v for k, v in row.items() if k != 'images'}

        images_array = row.get('images', [])
        saved_paths = []

        for img_idx, img_entry in enumerate(images_array):
            img_bytes = img_entry['bytes'] if isinstance(img_entry, dict) else img_entry
            img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

            sample_id = str(row.get('id', idx)).replace('/', '_')
            filename = f"{sample_id}_{img_idx}.png"
            filepath = image_dir_path / filename
            img.save(filepath, format='PNG')
            saved_paths.append(str(filepath))

        # Store as a single path string if there's only one image (common case),
        # otherwise store as a list to stay compatible with convert_to_parquet_hf.py
        record['image_path'] = saved_paths[0] if len(saved_paths) == 1 else saved_paths
        records.append(record)

        if (idx + 1) % 500 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Done! {len(records)} samples written to {output_json}")
    logger.info(f"Images saved to: {image_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert a Parquet dataset (embedded images) to JSON + image directory"
    )
    parser.add_argument('--input',     required=True, help='Path to input Parquet file')
    parser.add_argument('--output',    required=True, help='Path to output JSON file')
    parser.add_argument('--image_dir', required=True, help='Directory to save extracted images')
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        return

    convert_parquet_to_json(args.input, args.output, args.image_dir)


if __name__ == '__main__':
    main()
