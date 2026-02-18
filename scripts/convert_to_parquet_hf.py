#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert JSON dataset to Hugging Face Datasets format and save as Parquet
Following the standard approach of geometry3k
"""

import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from datasets import Image as ImageData
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_data_from_json(json_path: str, image_base_dir: str = None):
    """
    Generate data from JSON file

    Args:
        json_path: Path to JSON file
        image_base_dir: Optional base directory to resolve relative image paths

    Yields:
        dict: Dictionary containing images and other fields
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        # Get image path
        image_path = item['image_path']
        if isinstance(image_path, list):
            image_path = image_path[0]

        # Resolve relative paths
        if image_base_dir and not Path(image_path).is_absolute():
            image_path = str(Path(image_base_dir) / image_path)

        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Build output dictionary (use .get() for optional fields)
            output = {
                'id': item['id'],
                'dataset': item.get('dataset', ''),
                'split': item.get('split', 'train'),
                'question': item['question'],
                'answer': item['answer'],
                'images': [image],
                'problem': item.get('problem', ''),
            }
            
            # Add optional fields
            if 'has_overlay' in item:
                output['has_overlay'] = item['has_overlay']
            if 'overlay_position' in item:
                output['overlay_position'] = item['overlay_position']
            if 'is_50percent_mixed' in item:
                output['is_50percent_mixed'] = item['is_50percent_mixed']
            if 'image_path' in item:
                output['image_path'] = item['image_path'] if isinstance(item['image_path'], str) else item['image_path'][0]
            if 'choices' in item:
                output['choices'] = item['choices']
            if 'unit' in item:
                output['unit'] = item['unit']
            if 'precision' in item:
                output['precision'] = item['precision']
            if 'question_type' in item:
                output['question_type'] = item['question_type']
            
            yield output
            
        except Exception as e:
            logger.error(f"Failed to process {item.get('id', 'unknown')}: {e}")
            continue


def convert_json_to_hf_dataset(json_path: str, output_parquet_path: str, image_base_dir: str = None):
    """
    Convert JSON file to HF Dataset and save as Parquet

    Args:
        json_path: Path to input JSON file
        output_parquet_path: Path to output Parquet file
        image_base_dir: Optional base directory for resolving relative image paths
    """
    logger.info(f"Starting conversion: {json_path}")

    dataset = Dataset.from_generator(
        generate_data_from_json,
        gen_kwargs={"json_path": json_path, "image_base_dir": image_base_dir}
    )
    
    logger.info(f"  Created Dataset with {len(dataset)} samples")
    
    # Cast images column to proper type (using Sequence because images is a list)
    dataset = dataset.cast_column("images", Sequence(ImageData()))
    
    logger.info(f"  Casted images column to Sequence(ImageData) type")
    
    # Save as Parquet
    dataset.to_parquet(output_parquet_path)
    
    # Check file size
    json_size = Path(json_path).stat().st_size / (1024 * 1024)  # MB
    parquet_size = Path(output_parquet_path).stat().st_size / (1024 * 1024)  # MB
    
    logger.info(f"  Conversion completed!")
    logger.info(f"  JSON size: {json_size:.2f} MB")
    logger.info(f"  Parquet size: {parquet_size:.2f} MB")
    logger.info(f"  Output: {output_parquet_path}\n")
    
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert a JSON dataset file to Parquet (HuggingFace Datasets format)")
    parser.add_argument('--json', required=True, help='Path to input JSON file')
    parser.add_argument('--output', required=True, help='Path to output Parquet file')
    parser.add_argument('--image_base_dir', default=None,
                        help='Base directory for resolving relative image paths in the JSON (optional)')
    args = parser.parse_args()

    json_path = args.json
    parquet_path = args.output
    image_base_dir = args.image_base_dir

    if not Path(json_path).exists():
        logger.error(f"Input file not found: {json_path}")
        return

    if Path(parquet_path).exists():
        logger.info(f"Output already exists, delete it to re-convert: {parquet_path}")
        return

    logger.info("=" * 80)
    logger.info("Converting JSON → Parquet (HuggingFace Datasets)")
    logger.info("=" * 80)

    try:
        convert_json_to_hf_dataset(json_path, parquet_path, image_base_dir=image_base_dir)
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return

    size = Path(parquet_path).stat().st_size / (1024 * 1024)
    logger.info(f"Done! Output: {parquet_path} ({size:.2f} MB)")


if __name__ == '__main__':
    main()



