#!/usr/bin/env python3
"""
Vision Model Inference Script
Unified inference for test sets with embedded images
"""

import os
import json
import argparse
import sys
import signal
from typing import List, Dict
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI client not available")

interrupted = False
current_output_file = None
processed_data = []


def signal_handler(signum, frame):
    global interrupted, current_output_file, processed_data
    print(f"\nInterrupt received, saving progress...")
    interrupted = True
    
    if current_output_file and processed_data:
        temp_file = current_output_file.replace('.jsonl', '_interrupted.jsonl')
        with open(temp_file, 'w', encoding='utf-8') as f:
            for record in processed_data:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"Saved {len(processed_data)} records to {temp_file}")
    
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def pad_image_to_min_size(image: Image.Image, min_size: int = 224) -> Image.Image:
    """Pad image to minimum size if needed."""
    w, h = image.size
    
    if w >= min_size and h >= min_size:
        return image
    
    new_w = max(w, min_size)
    new_h = max(h, min_size)
    
    padded_img = Image.new('RGB', (new_w, new_h), (255, 255, 255))
    
    offset_x = (new_w - w) // 2
    offset_y = (new_h - h) // 2
    padded_img.paste(image, (offset_x, offset_y))
    
    return padded_img


def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class OCRInferenceEngine:
    """Vision model inference engine."""
    
    def __init__(self, api_base: str, api_key: str, model_name: str, 
                 temperature: float = 0.0, max_tokens: int = 2048, timeout: float = 300.0):
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        if not OPENAI_AVAILABLE:
            raise ImportError("Install openai package: pip install openai")
        
        self.client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout)
        self._test_connection()
    
    def _test_connection(self):
        try:
            models = self.client.models.list()
            if models.data:
                actual_model = models.data[0].id
                print(f"Connection successful, model: {actual_model}")
                self.model_name = actual_model
            else:
                print(f"Using specified model: {self.model_name}")
        except Exception as e:
            print(f"Connection test failed: {e}")
            raise
    
    def create_messages(self, question: str, has_overlay: bool = False, prompt_template: str = "simple") -> list:
        """Create messages for API call."""
        if prompt_template == "grpo":
            system_prompt = "You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \\boxed{}."
            if has_overlay:
                user_content = "Please analyze the image carefully and solve any problem or question shown in the image."
            else:
                user_content = f"Please analyze the image carefully and answer the following question:\n\n{question}"
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
        else:
            if has_overlay:
                user_content = "Please analyze the image carefully and solve any problem or question shown in the image. Give the final answer directly."
            else:
                user_content = f"Please analyze the image carefully and answer the following question:\n\n{question}\n\nGive the final answer directly."
            return [
                {"role": "user", "content": user_content}
            ]
    
    def inference_single(self, image: Image.Image, messages: list) -> str:
        """Single sample inference."""
        try:
            image = pad_image_to_min_size(image, min_size=224)
            image_base64 = image_to_base64(image)
            
            for msg in reversed(messages):
                if msg["role"] == "user":
                    if isinstance(msg["content"], str):
                        msg["content"] = [
                            {"type": "text", "text": msg["content"]},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                    ]
                    break
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Inference failed: {e}")
            return f"[ERROR: {str(e)}]"


class TestDataProcessor:
    """Test data processor for parquet files."""
    
    def load_test_data(self, parquet_file: str) -> List[Dict]:
        """Load test data from parquet with embedded images."""
        print(f"Loading test data from: {parquet_file}")
        
        try:
            from datasets import load_dataset
            dataset = load_dataset('parquet', data_files=parquet_file, split='train')
            print(f"Loaded {len(dataset)} records from parquet")
            
            processed_data = []
            for item in tqdm(dataset, desc="Processing test data"):
                try:
                    images = item.get('images', [])
                    if not images:
                        print(f"Warning: Skipping item {item.get('id')} - no images")
                        continue
                    
                    img_data = images[0]
                    if isinstance(img_data, dict):
                        img_bytes = img_data.get('bytes')
                        if img_bytes:
                            image = Image.open(BytesIO(img_bytes))
                        else:
                            print(f"Warning: Skipping item {item.get('id')} - no image bytes")
                            continue
                    else:
                        image = img_data
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    if max(image.width, image.height) > 2048:
                        image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                    
                    processed_data.append({
                        'id': item.get('id', f"unknown_{len(processed_data)}"),
                        'dataset': item['dataset'],
                        'split': item.get('split', 'unknown'),
                        'question': item['question'],
                        'answer': item['answer'],
                        'image': image,
                        'has_overlay': item.get('has_overlay', False),
                        'question_type': item.get('question_type', ''),
                        'answer_type': item.get('answer_type', ''),
                        'choices': item.get('choices', None),
                        'precision': item.get('precision', 2),
                        'category': item.get('category', ''),
                        'figure_id': item.get('figure_id', '')
                    })
                    
                except Exception as e:
                    print(f"Warning: Failed to process item: {e}")
                    continue
            
            print(f"Successfully processed {len(processed_data)} items")
            
            split_counts = {}
            for item in processed_data:
                split = item['split']
                split_counts[split] = split_counts.get(split, 0) + 1
            
            print(f"\nData distribution by split:")
            for split, count in sorted(split_counts.items()):
                print(f"  - {split}: {count} samples")
            
            return processed_data
            
        except Exception as e:
            print(f"ERROR: Failed to load parquet file: {e}")
            raise


def process_single_item(inference_engine: OCRInferenceEngine, item: Dict, item_index: int, prompt_template: str = "simple") -> Dict:
    """Process single inference item."""
    try:
        messages = inference_engine.create_messages(
            item['question'], 
            item['has_overlay'],
            prompt_template=prompt_template
        )
        
        model_response = inference_engine.inference_single(
            item['image'], 
            messages
        )
        
        result = {
            'id': item['id'],
            'dataset': item['dataset'],
            'split': item['split'],
            'question': item['question'],
            'answer': item['answer'],
            'has_overlay': item['has_overlay'],
            'question_type': item.get('question_type', ''),
            'answer_type': item.get('answer_type', ''),
            'choices': item.get('choices', None),
            'precision': item.get('precision', 2),
            'category': item.get('category', ''),
            'figure_id': item.get('figure_id', ''),
            'model_answer': [model_response],
            'model_response': model_response,
            'inference_method': 'vllm_api',
            'item_index': item_index
        }
        
        return result
        
    except Exception as e:
        print(f"Warning: Failed to process item {item_index+1}: {e}")
        return {
            'id': item.get('id', f'error_{item_index}'),
            'dataset': item.get('dataset', 'unknown'),
            'split': item.get('split', 'unknown'),
            'question': item.get('question', ''),
            'answer': item.get('answer', ''),
            'has_overlay': item.get('has_overlay', False),
            'category': item.get('category', ''),
            'figure_id': item.get('figure_id', ''),
            'model_answer': [f"[ERROR: {str(e)}]"],
            'model_response': f"[ERROR: {str(e)}]",
            'inference_method': 'vllm_api_error',
            'item_index': item_index
        }


def run_inference_on_dataset(
    inference_engine: OCRInferenceEngine,
    test_data: List[Dict],
    output_file: str,
    start_idx: int = 0,
    max_workers: int = 8,
    prompt_template: str = "simple"
) -> List[Dict]:
    """Run inference on dataset."""
    global processed_data, interrupted
    
    results = []
    
    if start_idx > 0:
        results = processed_data.copy()
    
    print(f"Starting inference, total {len(test_data)} items")
    if start_idx > 0:
        print(f"   Resuming from item {start_idx + 1}")
    
    pbar = tqdm(
        total=len(test_data),
        initial=start_idx,
        desc="Inference progress"
    )
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {}
        for i, item in enumerate(test_data[start_idx:], start_idx):
            if interrupted:
                break
            future = executor.submit(process_single_item, inference_engine, item, i, prompt_template)
            future_to_index[future] = i
        
        results_dict = {}
        
        for future in as_completed(future_to_index):
            if interrupted:
                executor.shutdown(wait=False)
                break
            
            try:
                result = future.result()
                index = future_to_index[future]
                results_dict[index] = result
                
                pbar.update(1)
                current_count = len(results_dict)
                current_accuracy = len([r for r in results_dict.values() if 'ERROR' not in str(r.get('model_response', ''))]) / current_count * 100 if current_count else 0
                pbar.set_description(f"Inference progress (completed: {current_count}, success rate: {current_accuracy:.1f}%)")
                
            except Exception as e:
                index = future_to_index[future]
                print(f"Failed to process item {index+1}: {e}")
                continue
    
    pbar.close()
    
    print("\nSorting and saving results...")
    sorted_indices = sorted(results_dict.keys())
    
    with open(output_file, 'a', encoding='utf-8') as f:
        for idx in sorted_indices:
            result = results_dict[idx]
            results.append(result)
            processed_data.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(results)} results")
    
    return results


def main():
    global current_output_file, processed_data
    
    parser = argparse.ArgumentParser(description="Vision Model Inference Script")
    
    parser.add_argument("--api_base", type=str, default="http://localhost:8001/v1",
                       help="vLLM server address")
    parser.add_argument("--api_key", type=str, default="EMPTY", help="API key")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    
    parser.add_argument("--test_data", type=str, required=True,
                       help="Test data parquet file path")
    
    parser.add_argument("--output_file", type=str, required=True,
                       help="Output file path")
    
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Maximum generation length")
    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    parser.add_argument("--max_workers", type=int, default=8, help="Number of concurrent workers")
    parser.add_argument("--prompt_template", type=str, default="simple", choices=["simple", "grpo"],
                       help="Prompt template: 'simple' or 'grpo'")
    
    args = parser.parse_args()
    
    current_output_file = args.output_file
    
    print("=" * 70)
    print("Vision Model Inference")
    print("=" * 70)
    print(f"Server: {args.api_base}")
    print(f"Model: {args.model_name}")
    print(f"Test data: {args.test_data}")
    print(f"Output: {args.output_file}")
    print(f"Prompt: {args.prompt_template}")
    print("=" * 70)
    
    if not os.path.exists(args.test_data):
        print(f"Test data file does not exist: {args.test_data}")
        sys.exit(1)
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    start_idx = 0
    interrupted_file = args.output_file.replace('.jsonl', '_interrupted.jsonl')
    
    if os.path.exists(interrupted_file):
        print(f"Found interrupted file: {interrupted_file}")
        with open(interrupted_file, 'r', encoding='utf-8') as f:
            processed_data = [json.loads(line) for line in f if line.strip()]
            start_idx = len(processed_data)
        
        print(f"   Auto-resuming from {start_idx} processed records")
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for result in processed_data:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        os.remove(interrupted_file)
        
    elif os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        print(f"Found existing output file: {args.output_file}")
        with open(args.output_file, 'r', encoding='utf-8') as f:
            processed_data = [json.loads(line) for line in f if line.strip()]
            start_idx = len(processed_data)
        
        print(f"   Auto-resuming from {start_idx} processed records")
        
    else:
        open(args.output_file, 'w').close()
        print(f"Starting fresh inference")
    
    try:
        inference_engine = OCRInferenceEngine(
            api_base=args.api_base,
            api_key=args.api_key,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout=args.timeout
        )
        
        data_processor = TestDataProcessor()
        test_data = data_processor.load_test_data(args.test_data)
        
        if not test_data:
            print("No valid test data")
            sys.exit(1)
        
        results = run_inference_on_dataset(
            inference_engine,
            test_data,
            args.output_file,
            start_idx,
            args.max_workers,
            args.prompt_template
        )
        
        if os.path.exists(interrupted_file):
            os.remove(interrupted_file)
        
        print(f"\nInference completed!")
        print(f"Results saved to: {args.output_file}")
        print(f"Total processed: {len(results)} records")
        
    except Exception as e:
        print(f"ERROR: Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
