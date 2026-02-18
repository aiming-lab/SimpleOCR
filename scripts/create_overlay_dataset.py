#!/usr/bin/env python3
"""
Create a dataset variant with text overlaid on images.
- Overlay the question text onto the image
- Replace the question field with a simple prompt
"""
import image_text_overlay

import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from tqdm import tqdm

def extract_question_text(question_str):
    """Extract the actual text from a question string, removing tags like <image>."""
    # Remove <image> and <image1> style tags
    text = re.sub(r'<image\d*>', '', question_str)
    text = text.strip()
    return text

def add_text_to_image_bytes(image_bytes, text):
    """Overlay text onto an image using the verl utility."""
    # Load the image
    img = Image.open(io.BytesIO(image_bytes))
    
    # Convert to RGB if necessary
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Add text using the verl utility
    new_img = image_text_overlay.add_text_to_image(img, text, seed=None)
    
    # Convert the new image back to bytes
    img_byte_arr = io.BytesIO()
    new_img.save(img_byte_arr, format='PNG')
    return img_byte_arr.getvalue()

def create_overlay_dataset(input_path, output_path):
    """Create a dataset with text overlaid on images."""
    print(f"Loading dataset: {input_path}")
    df = pd.read_parquet(input_path)
    
    print(f"Total samples: {len(df)}")
    
    # Build new dataset records
    new_records = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
        # Extract the question text
        question_text = extract_question_text(row['question'])
        
        # If question_text is empty or too short, keep the row as-is
        if len(question_text.strip()) < 5:
            # Skip or keep original
            new_records.append(row.to_dict())
            continue
        
        # Process images
        images_array = row['images']
        new_images = []
        
        for img_dict in images_array:
            img_bytes = img_dict['bytes']
            
            # Overlay text onto the image
            try:
                new_img_bytes = add_text_to_image_bytes(img_bytes, question_text)
                new_images.append({'bytes': new_img_bytes, 'path': None})
            except Exception as e:
                print(f"\nWarning: failed to process image for sample {idx}: {e}")
                # Fall back to original image
                new_images.append(img_dict)
        
        # Build the new record
        new_record = row.to_dict()
        new_record['images'] = np.array(new_images, dtype=object)
        
        # Replace the question field with a simple prompt.
        # Use the problem field if it exists and is non-empty, otherwise use a default prompt.
        if row['problem'] and str(row['problem']).strip():
            new_record['question'] = row['problem']
        else:
            new_record['question'] = '<image>Please answer the question shown in the image.\n<image1>'
        
        # Store the original question text in the problem field
        new_record['problem'] = question_text
        
        new_records.append(new_record)
    
    # Create new DataFrame
    print("\nCreating new DataFrame...")
    new_df = pd.DataFrame(new_records)
    
    # Cast columns to appropriate dtypes (only for columns that actually exist)
    dtype_map = {
        'id': 'object',
        'images': 'object',
        'dataset': 'object',
        'split': 'object',
        'question': 'object',
        'answer': 'object',
        'has_overlay': 'bool',
        'problem': 'object',
        'question_type': 'object',
        'answer_type': 'object',
        'choices': 'object',
        'precision': 'float64',
        'category': 'object',
        'figure_id': 'object',
    }
    cast_map = {col: dtype for col, dtype in dtype_map.items() if col in new_df.columns}
    new_df = new_df.astype(cast_map)
    
    # Save
    print(f"\nSaving to: {output_path}")
    new_df.to_parquet(output_path, engine='pyarrow', index=False)
    
    print(f"\nDone! Processed {len(new_df)} samples in total.")
    print(f"Output file: {output_path}")
    
    return new_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Overlay question text onto images in a Parquet dataset")
    parser.add_argument('--input', required=True, help='Path to input Parquet file')
    parser.add_argument('--output', required=True, help='Path to output Parquet file')
    args = parser.parse_args()

    create_overlay_dataset(args.input, args.output)
