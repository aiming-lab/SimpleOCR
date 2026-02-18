#!/usr/bin/env python3
"""
Vision Model Evaluation Script
Supports Math-Verify + LLM fallback evaluation
"""

import os
import json
import argparse
import signal
import sys
import re
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from Levenshtein import distance as levenshtein_distance
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

interrupted = False
current_output_file = None
processed_results = []


def signal_handler(signum, frame):
    global interrupted, current_output_file, processed_results
    print(f"\nInterrupt received, saving progress...")
    interrupted = True
    
    if current_output_file and processed_results:
        temp_file = current_output_file.replace('.jsonl', '_interrupted.jsonl')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                for record in processed_results:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Saved {len(processed_results)} records to {temp_file}")
        except Exception as e:
            print(f"Error saving: {e}")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    from openai import OpenAI, AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI not available, install: pip install openai")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available")

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
    MATH_VERIFY_AVAILABLE = True
    print("Math-Verify loaded successfully")
except ImportError:
    MATH_VERIFY_AVAILABLE = False
    print("Warning: Math-Verify not available!")
    print("Install: pip install math-verify")


def get_most_similar(prediction, choices):
    """Use Levenshtein distance to find most similar choice."""
    distances = [levenshtein_distance(str(prediction), str(choice)) for choice in choices]
    return choices[distances.index(min(distances))]


def normalize_answer(extraction, question_type="", answer_type="", choices=None, precision=2):
    """Normalize extracted answer following MathVista official logic."""
    if not extraction or extraction == "":
        return None
        
    extraction = str(extraction).strip()
    
    if question_type == 'multi_choice' and choices:
        letter_match = re.findall(r'\(?([a-zA-Z])\)?', extraction)
        if letter_match:
            extraction = letter_match[0].upper()
        
        sequential_chars = [chr(ord('A') + i) for i in range(len(choices))]
        
        if extraction in sequential_chars:
            return choices[sequential_chars.index(extraction)]
        
        if extraction in choices:
            return extraction
        
        return get_most_similar(extraction, choices)
    
    if answer_type == 'integer':
        try:
            return str(int(float(extraction)))
        except:
            return None
    
    if answer_type == 'float':
        try:
            return str(round(float(extraction), int(precision)))
        except:
            return None
    
    if answer_type == 'list':
        try:
            return str(extraction)
        except:
            return None
    
    return extraction


class SimpleEvaluator:
    """Evaluator with Math-Verify and LLM fallback."""
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-72B-Instruct", 
                 use_vllm: bool = False, use_azure: bool = False,
                 api_base: str = "http://localhost:8001/v1",
                 azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, azure_deployment: str = None):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_azure = use_azure
        self.eval_mode = "hybrid"
        
        if MATH_VERIFY_AVAILABLE:
            self.verify_func = math_metric(
                gold_extraction_target=(LatexExtractionConfig(),),
                pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
            )
        else:
            self.verify_func = None
        
        if use_azure and OPENAI_AVAILABLE:
            if not azure_api_key or not azure_endpoint:
                raise RuntimeError("Azure OpenAI requires api_key and endpoint!")
            
            self.client = AzureOpenAI(
                api_key=azure_api_key,
                azure_endpoint=azure_endpoint,
                api_version=azure_api_version or "2024-08-01-preview"
            )
            self.model_name = azure_deployment or model_name
            print(f"Using Azure OpenAI")
            print(f"  Endpoint: {azure_endpoint}")
            print(f"  Deployment: {self.model_name}")
        elif use_vllm and OPENAI_AVAILABLE:
            self.client = OpenAI(api_key="EMPTY", base_url=api_base)
            self._test_vllm_connection()
            print(f"Using vLLM: {model_name}")
        elif TRANSFORMERS_AVAILABLE:
            self._load_local_model()
            print(f"Using local model: {model_name}")
        else:
            raise RuntimeError("No LLM available! Install openai or transformers")

    def set_eval_mode(self, eval_mode: str):
        """Set evaluation mode: 'hybrid' or 'llm_only'."""
        if eval_mode not in ("hybrid", "llm_only"):
            raise ValueError(f"Unknown eval_mode: {eval_mode}")
        self.eval_mode = eval_mode
    
    def _test_vllm_connection(self):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=5
            )
            print("vLLM server connected")
        except Exception as e:
            raise RuntimeError(f"vLLM connection failed: {e}")
    
    def _load_local_model(self):
        print(f"Loading local model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        print("Local model loaded")
    
    def _extract_boxed_answer(self, text: str) -> str:
        """Extract answer from \\boxed{} or <answer> tags."""
        if not text:
            return text
        
        answer_pattern = r'<answer>(.*?)</answer>'
        answer_matches = re.findall(answer_pattern, text, re.DOTALL | re.IGNORECASE)
        if answer_matches:
            answer = answer_matches[-1].strip()
            return answer
        
        boxed_start = text.find(r'\boxed{')
        if boxed_start != -1:
            start_pos = boxed_start + 7
            brace_count = 1
            pos = start_pos
            
            while pos < len(text) and brace_count > 0:
                if text[pos] == '{':
                    brace_count += 1
                elif text[pos] == '}':
                    brace_count -= 1
                pos += 1
            
            if brace_count == 0:
                answer = text[start_pos:pos-1].strip()
                answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
                answer = re.sub(r'\\mathrm\{([^}]+)\}', r'\1', answer)
                answer = re.sub(r'\\,', ' ', answer)
                answer = answer.replace('$', '').replace('\\', '').strip()
                return answer
        
        return text
    
    def _get_most_similar(self, prediction: str, choices: list) -> str:
        """Use Levenshtein distance to find the most similar choice."""
        if not choices:
            return prediction
        
        distances = [levenshtein_distance(prediction, choice) for choice in choices]
        min_idx = distances.index(min(distances))
        return choices[min_idx]
    
    def _normalize_choice_answer(self, extraction: str, ground_truth: str, choices: list) -> tuple:
        """Normalize multiple choice answers for comparison."""
        if not choices:
            return extraction, ground_truth
        
        extraction = str(extraction).strip()
        ground_truth = str(ground_truth).strip()
        
        sequential_letters = [chr(ord('A') + i) for i in range(len(choices))]
        
        letter_pattern = r'\(([a-zA-Z])\)'
        letter_match = re.findall(letter_pattern, extraction)
        if letter_match:
            extraction = letter_match[0].upper()
        
        normalized_extraction = extraction
        if extraction.upper() in sequential_letters:
            option_index = sequential_letters.index(extraction.upper())
            normalized_extraction = choices[option_index]
        else:
            normalized_extraction = self._get_most_similar(extraction, choices)
        
        normalized_gt = ground_truth
        if ground_truth.upper() in sequential_letters:
            option_index = sequential_letters.index(ground_truth.upper())
            normalized_gt = choices[option_index]
        else:
            normalized_gt = self._get_most_similar(ground_truth, choices)
        
        return normalized_extraction, normalized_gt
    
    def evaluate(self, model_output: str, ground_truth: str, question: str = "", 
                 question_type: str = "", answer_type: str = "", choices: list = None, 
                 precision: int = 1, dataset: str = "", category: str = "", figure_id: str = "") -> Dict:
        """Hybrid evaluation strategy."""
        result = {
            "score": 0.0,
            "method": "none",
            "extraction": "",
            "prediction": "",
            "invalid_ground_truth": False,
            "math_verify_passed": False
        }

        if ground_truth is None or str(ground_truth).strip() == "":
            result["invalid_ground_truth"] = True
            result["method"] = "invalid_gt"
            return result
        
        if dataset.lower() == "hallusionbench":
            return self._evaluate_hallusionbench(model_output, ground_truth, question, category, figure_id)
        
        extracted_answer = self._extract_boxed_answer(model_output)

        if question_type == 'multi_choice' and choices:
            normalized_extraction, normalized_gt = self._normalize_choice_answer(
                extracted_answer, ground_truth, choices
            )
            
            if normalized_extraction.strip().lower() == normalized_gt.strip().lower():
                result["score"] = 1.0
                result["method"] = "choice_normalized_match"
                result["extraction"] = normalized_extraction
                result["prediction"] = normalized_gt
                return result
            
            extracted_answer = normalized_extraction
            ground_truth = normalized_gt
        
        if getattr(self, "eval_mode", "hybrid") == "llm_only":
            llm_score = self._llm_judge_direct(model_output, ground_truth, question)
            result["score"] = float(llm_score) if llm_score >= 0 else 0.0
            result["method"] = "llm_judge" if llm_score == 1 else "no_match"
            result["extraction"] = extracted_answer
            return result
        
        if dataset.lower() == "ocrbench":
            gt_lower = str(ground_truth).strip().lower()
            ext_lower = str(extracted_answer).strip().lower()
            if gt_lower == ext_lower:
                result["score"] = 1.0
                result["method"] = "ocr_case_insensitive_match"
                result["extraction"] = extracted_answer
                result["prediction"] = ground_truth
                return result
        
        if MATH_VERIFY_AVAILABLE and self.verify_func:
            try:
                ground_truth_boxed = f"\\boxed{{{ground_truth}}}"
                extracted_boxed = f"\\boxed{{{extracted_answer}}}"
                ret_score, _ = self.verify_func([ground_truth_boxed], [extracted_boxed])
                
                if ret_score > 0:
                    result["score"] = 1.0
                    result["method"] = "math_verify"
                    result["math_verify_passed"] = True
                    result["extraction"] = extracted_answer
                    return result
            except Exception:
                pass
        
        llm_score = self._llm_judge_direct(model_output, ground_truth, question)
        result["score"] = float(llm_score) if llm_score >= 0 else 0.0
        result["method"] = "llm_judge" if llm_score == 1 else "no_match"
        result["extraction"] = extracted_answer
        
        return result
    
    def _evaluate_hallusionbench(self, model_output: str, ground_truth: str, question: str = "", 
                                  category: str = "", figure_id: str = "") -> Dict:
        """HallusionBench official evaluation logic."""
        result = {
            "score": 0.0,
            "method": "hallusionbench",
            "extraction": "",
            "prediction": "",
            "invalid_ground_truth": False,
            "math_verify_passed": False
        }
        
        extracted_answer = self._extract_boxed_answer(model_output).strip()
        
        pred_value = None
        extracted_lower = extracted_answer.lower()
        
        if extracted_answer in ['0', '1', '2']:
            pred_value = extracted_answer
        elif extracted_lower in ['yes', 'no']:
            pred_value = '1' if extracted_lower == 'yes' else '0'
        elif extracted_lower in ['true', 'false']:
            pred_value = '1' if extracted_lower == 'true' else '0'
        elif 'uncertain' in extracted_lower or 'unclear' in extracted_lower or 'unsure' in extracted_lower:
            pred_value = '2'
        elif 'yes' in extracted_lower and 'no' not in extracted_lower:
            pred_value = '1'
        elif 'no' in extracted_lower and 'yes' not in extracted_lower:
            pred_value = '0'
        else:
            pred_value = '0'
        
        gt_value = str(ground_truth).strip()
        
        if pred_value == gt_value:
            result["score"] = 1.0
            result["method"] = "hallusionbench_match"
        else:
            result["score"] = 0.0
            result["method"] = "hallusionbench_no_match"
        
        result["extraction"] = extracted_answer
        result["prediction"] = pred_value
        
        return result
    
    def _llm_judge_direct(self, model_output: str, ground_truth: str, question: str = "") -> int:
        """LLM judge for general datasets."""
        
        prompt = f"""Evaluate if the model's answer is correct.

Question: {question}
Correct Answer: {ground_truth}
Model Response: {model_output}

Rules:
- Ignore formatting/LaTeX/units (e.g., "1.2cm" = "1.2")
- Multiple choice: only letter matters (A/B/C/D/E)
- Numbers: allow ±1% tolerance
- Focus on final answer, not reasoning

Reply with ONLY one character: "1" if correct, "0" if incorrect.
No explanation, no parentheses, just the digit.

Answer:"""
        
        try:
            if self.use_vllm or self.use_azure:
                api_params = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                }
                if self.use_azure:
                    api_params["max_completion_tokens"] = 2048
                else:
                    api_params["temperature"] = 0.0
                    api_params["max_tokens"] = 2048
                
                response = self.client.chat.completions.create(**api_params)
                
                message = response.choices[0].message
                result = getattr(message, 'content', None)
                
                if result is None or result == "":
                    refusal = getattr(message, 'refusal', None)
                    if refusal:
                        print(f"Warning: LLM judge refused to answer: {refusal}")
                    result = ""
                else:
                    result = result.strip()
            else:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = self.tokenizer([text], return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=2048, temperature=0.0, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
                result = self.tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
            
            first_char = result.strip()[0] if result.strip() else ""
            if first_char == "1":
                return 1
            elif first_char == "0":
                return 0
            else:
                if "1" in result:
                    return 1
                elif "0" in result:
                    return 0
                else:
                    return 0
        
        except Exception as e:
            print(f"Warning: LLM judge failed: {e}")
            return 0


class VisionEvaluator:
    """Main evaluation orchestrator with detailed statistics."""
    
    def __init__(self, evaluator_model: str = "Qwen/Qwen2.5-72B-Instruct",
                 use_vllm: bool = False, use_azure: bool = False,
                 api_base: str = "http://localhost:8001/v1",
                 azure_api_key: str = None, azure_endpoint: str = None,
                 azure_api_version: str = None, azure_deployment: str = None,
                 only_llm_judge: bool = False,
                 max_workers: int = 1):
        
        self.evaluator = SimpleEvaluator(
            model_name=evaluator_model,
            use_vllm=use_vllm,
            use_azure=use_azure,
            api_base=api_base,
            azure_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment
        )
        if only_llm_judge:
            self.evaluator.set_eval_mode("llm_only")
        
        self.max_workers = max_workers
        self.stats_lock = threading.Lock()
        
        print(f"\n{'='*60}")
        print(f"Hybrid Evaluator Initialized")
        print(f"{'='*60}")
        print(f"  Model: {evaluator_model}")
        print(f"  Max Workers: {max_workers}")
        print(f"  Strategies:")
        print(f"    - HallusionBench: GPT-4 style judge (0/1/2)")
        if only_llm_judge:
            print(f"    - Others: LLM judge only")
        else:
            print(f"    - Others: Math-Verify + LLM fallback")
        print(f"{'='*60}\n")
    
    def load_results(self, results_path: str, dataset_filter: str = None) -> List[Dict]:
        """Load inference results with optional dataset filtering."""
        print(f"Loading: {results_path}")
        results = []
        with open(results_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if line.strip():
                        data = json.loads(line.strip())
                        if dataset_filter:
                            if data.get('dataset', '').lower() == dataset_filter.lower():
                                results.append(data)
                        else:
                            results.append(data)
                except Exception as e:
                    print(f"Warning: Line {line_num} parse error: {e}")
        
        if dataset_filter:
            print(f"Loaded {len(results)} results (filtered by dataset='{dataset_filter}')")
        else:
            print(f"Loaded {len(results)} results")
        return results
    
    def evaluate_single(self, result: Dict) -> Dict:
        """Evaluate single result."""
        model_output = result.get('model_response', '') or result.get('model_answer', '')
        if isinstance(model_output, list):
            model_output = model_output[0] if model_output else ''
        
        correct_answer = str(result.get('answer', ''))
        question = str(result.get('question', ''))
        dataset = result.get('dataset', '')
        question_type = result.get('question_type', '')
        answer_type = result.get('answer_type', '')
        choices = result.get('choices', None)
        precision = result.get('precision', 1)
        category = result.get('category', '')
        figure_id = result.get('figure_id', '')
        
        evaluation = self.evaluator.evaluate(
            str(model_output), correct_answer, question,
            question_type=question_type,
            answer_type=answer_type,
            choices=choices,
            precision=precision,
            dataset=dataset,
            category=category,
            figure_id=figure_id
        )
        
        return {
            **result,
            "evaluation": evaluation,
            "score": evaluation["score"]
        }
    
    def evaluate_dataset(self, results: List[Dict], start_idx: int = 0) -> tuple:
        """Evaluate entire dataset with detailed statistics."""
        global interrupted, processed_results
        
        print(f"Starting evaluation...")
        if start_idx > 0:
            print(f"   Resuming from item {start_idx + 1}")
        
        evaluated_results = processed_results.copy() if start_idx > 0 else []
        
        stats = {
            "total": len(results),
            "correct": 0,
            "average_score": 0.0,
            "math_verify_count": 0,
            "llm_count": 0,
            "invalid_gt_count": 0,
            "by_method": defaultdict(int),
            "by_dataset": defaultdict(lambda: {"total": 0, "correct": 0, "average_score": 0.0}),
            "by_split": defaultdict(lambda: {
                "total": 0,
                "correct": 0,
                "average_score": 0.0,
                "invalid_gt_count": 0,
                "by_dataset": defaultdict(lambda: {"total": 0, "correct": 0, "average_score": 0.0})
            })
        }
        
        if start_idx > 0:
            for result in evaluated_results:
                self._update_stats(result, stats)
        
        remaining = results[start_idx:]
        if not remaining:
            return evaluated_results, stats
        
        pbar = tqdm(total=len(results), initial=start_idx, desc="Evaluating")
        
        if self.max_workers > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_idx = {
                    executor.submit(self.evaluate_single, result): (start_idx + i, result)
                    for i, result in enumerate(remaining)
                }
                
                for future in as_completed(future_to_idx):
                    if interrupted:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    idx, result = future_to_idx[future]
                    try:
                        evaluated = future.result()
                        
                        with self.stats_lock:
                            evaluated_results.append(evaluated)
                            processed_results.append(evaluated)
                            self._update_stats(evaluated, stats)
                        
                        pbar.update(1)
                        
                        if evaluated_results:
                            acc = stats["correct"] / len(evaluated_results) * 100
                            pbar.set_description(f"Evaluating (Acc: {acc:.1f}%)")
                    
                    except Exception as e:
                        print(f"Warning: Error at {idx}: {e}")
                        pbar.update(1)
                        continue
        else:
            for i, result in enumerate(remaining):
                if interrupted:
                    break
                
                try:
                    evaluated = self.evaluate_single(result)
                    evaluated_results.append(evaluated)
                    processed_results.append(evaluated)
                    
                    self._update_stats(evaluated, stats)
                    
                    pbar.update(1)
                    
                    if evaluated_results:
                        acc = stats["correct"] / len(evaluated_results) * 100
                        pbar.set_description(f"Evaluating (Acc: {acc:.1f}%)")
                
                except Exception as e:
                    print(f"Warning: Error at {start_idx + i}: {e}")
                    continue
        
        pbar.close()
        
        if evaluated_results:
            total = len(evaluated_results)
            stats["accuracy"] = stats["correct"] / total
            stats["average_score"] = stats["average_score"] / total
            
            for ds in stats["by_dataset"]:
                ds_total = stats["by_dataset"][ds]["total"]
                if ds_total > 0:
                    stats["by_dataset"][ds]["accuracy"] = stats["by_dataset"][ds]["correct"] / ds_total
                    stats["by_dataset"][ds]["average_score"] /= ds_total
            
            for split in stats["by_split"]:
                split_total = stats["by_split"][split]["total"]
                if split_total > 0:
                    stats["by_split"][split]["accuracy"] = stats["by_split"][split]["correct"] / split_total
                    stats["by_split"][split]["average_score"] /= split_total
                    
                    for ds in stats["by_split"][split]["by_dataset"]:
                        ds_total = stats["by_split"][split]["by_dataset"][ds]["total"]
                        if ds_total > 0:
                            stats["by_split"][split]["by_dataset"][ds]["accuracy"] = \
                                stats["by_split"][split]["by_dataset"][ds]["correct"] / ds_total
                            stats["by_split"][split]["by_dataset"][ds]["average_score"] /= ds_total
        
        return evaluated_results, stats
    
    def _update_stats(self, result: Dict, stats: Dict):
        """Update statistics with single result."""
        dataset = result.get("dataset", "unknown")
        split = result.get("split", "unknown")
        evaluation = result.get("evaluation", {})
        score = result.get("score", 0.0)
        
        if evaluation.get("invalid_ground_truth"):
            stats["invalid_gt_count"] += 1
            stats["by_split"][split]["invalid_gt_count"] += 1
            stats["by_dataset"][dataset]["total"] += 1
            stats["by_split"][split]["total"] += 1
            stats["by_split"][split]["by_dataset"][dataset]["total"] += 1
            return

        stats["average_score"] += score
        if score >= 1.0:
            stats["correct"] += 1
        
        method = evaluation.get("method", "none")
        stats["by_method"][method] += 1
        
        if evaluation.get("math_verify_passed"):
            stats["math_verify_count"] += 1
        if method == "llm_judge":
            stats["llm_count"] += 1
        
        stats["by_dataset"][dataset]["total"] += 1
        stats["by_dataset"][dataset]["average_score"] += score
        if score >= 1.0:
            stats["by_dataset"][dataset]["correct"] += 1
        
        stats["by_split"][split]["total"] += 1
        stats["by_split"][split]["average_score"] += score
        if score >= 1.0:
            stats["by_split"][split]["correct"] += 1
        
        stats["by_split"][split]["by_dataset"][dataset]["total"] += 1
        stats["by_split"][split]["by_dataset"][dataset]["average_score"] += score
        if score >= 1.0:
            stats["by_split"][split]["by_dataset"][dataset]["correct"] += 1
    
    def save_results(self, evaluated_results: List[Dict], stats: Dict, output_path: str):
        """Save results and statistics."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in evaluated_results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        stats_path = output_path.replace('.jsonl', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Results: {output_path}")
        print(f"Stats: {stats_path}")
    
    def print_summary(self, stats: Dict):
        """Print evaluation summary."""
        print("\n" + "="*80)
        print("Evaluation Summary")
        print("="*80)
        
        total = stats['total']
        print(f"\nOverall Metrics:")
        print(f"  Total samples: {total}")
        print(f"  Accuracy: {stats.get('accuracy', 0):.2%} ({stats['correct']}/{total})")
        print(f"  Average score: {stats.get('average_score', 0):.4f}")
        
        print(f"\nEvaluation Methods:")
        print(f"  Math-Verify: {stats['math_verify_count']} ({stats['math_verify_count']/total:.1%})")
        print(f"  LLM Judge: {stats['llm_count']} ({stats['llm_count']/total:.1%})")
        
        if stats.get("invalid_gt_count", 0) > 0:
            print(f"  Invalid GT: {stats['invalid_gt_count']} ({stats['invalid_gt_count']/total:.1%})")
        
        if stats["by_split"]:
            print("\n" + "="*80)
            print("Results by Split")
            print("="*80)
            
            for split in sorted(stats["by_split"].keys()):
                split_stats = stats["by_split"][split]
                if split_stats["total"] > 0:
                    print(f"\n{split.upper()}")
                    print(f"   Total: {split_stats['total']} samples")
                    print(f"   Accuracy: {split_stats.get('accuracy', 0):.2%} ({split_stats['correct']}/{split_stats['total']})")
                    print(f"   Avg Score: {split_stats.get('average_score', 0):.4f}")
                    
                    if split_stats["by_dataset"]:
                        print(f"   Datasets:")
                        for ds in sorted(split_stats["by_dataset"].keys()):
                            ds_stats = split_stats["by_dataset"][ds]
                            if ds_stats["total"] > 0:
                                print(f"     - {ds:20s} "
                                      f"Acc={ds_stats.get('accuracy', 0):.2%} "
                                      f"({ds_stats['correct']}/{ds_stats['total']})  "
                                      f"Score={ds_stats.get('average_score', 0):.4f}")


def main():
    global current_output_file, processed_results
    
    parser = argparse.ArgumentParser(description="Vision Model Evaluation with Math-Verify + LLM")
    parser.add_argument("--results_path", required=True, help="Path to inference results")
    parser.add_argument("--output_path", required=True, help="Path to output")
    parser.add_argument("--evaluator_model", default="Qwen/Qwen2.5-72B-Instruct", 
                       help="LLM for fallback evaluation")
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM")
    parser.add_argument("--use_azure", action="store_true", help="Use Azure OpenAI")
    parser.add_argument("--api_base", default="http://localhost:8001/v1", help="vLLM server")
    parser.add_argument("--azure_api_key", default=None, help="Azure OpenAI API key")
    parser.add_argument("--azure_endpoint", default=None, help="Azure OpenAI endpoint")
    parser.add_argument("--azure_deployment", default=None, help="Azure deployment name")
    parser.add_argument("--azure_api_version", default=None, help="Azure API version")
    parser.add_argument("--dataset_filter", default=None, 
                       help="Filter by dataset name (e.g., 'MathVista', 'OCRBench')")
    parser.add_argument("--only_llm_judge", action="store_true",
                       help="Use only LLM judge for non-HallusionBench datasets")
    parser.add_argument("--max_workers", type=int, default=1,
                       help="Number of concurrent workers for evaluation (default: 1)")
    
    args = parser.parse_args()
    current_output_file = args.output_path
    
    print(f"\n{'='*70}")
    print(f"Vision Model Evaluation")
    print(f"{'='*70}")
    print(f"Input: {args.results_path}")
    print(f"Output: {args.output_path}")
    print(f"Model: {args.evaluator_model}")
    print(f"vLLM: {'Enabled' if args.use_vllm else 'Disabled'}")
    print(f"Azure OpenAI: {'Enabled' if args.use_azure else 'Disabled'}")
    if args.dataset_filter:
        print(f"Dataset Filter: {args.dataset_filter}")
    if args.only_llm_judge:
        print("Eval Mode: only_llm_judge")
    print(f"{'='*70}\n")
    
    start_idx = 0
    interrupted_file = args.output_path.replace('.jsonl', '_interrupted.jsonl')
    
    if os.path.exists(interrupted_file):
        print(f"\nFound interrupted file: {interrupted_file}")
        try:
            with open(interrupted_file, 'r', encoding='utf-8') as f:
                processed_results = [json.loads(line) for line in f if line.strip()]
                start_idx = len(processed_results)
            
            print(f"Already evaluated: {start_idx} items")
            response = input("Continue? (y/n): ").lower()
            if response != 'y':
                start_idx = 0
                processed_results = []
                os.remove(interrupted_file)
        except Exception as e:
            print(f"Warning: Failed to read interrupted file: {e}")
            start_idx = 0
            processed_results = []
    
    try:
        model_name = args.azure_deployment if args.use_azure and args.azure_deployment else args.evaluator_model
        
        evaluator = VisionEvaluator(
            evaluator_model=model_name,
            use_vllm=args.use_vllm,
            use_azure=args.use_azure,
            api_base=args.api_base,
            azure_api_key=args.azure_api_key,
            azure_endpoint=args.azure_endpoint,
            azure_api_version=args.azure_api_version,
            azure_deployment=args.azure_deployment,
            only_llm_judge=args.only_llm_judge,
            max_workers=args.max_workers
        )
        
        results = evaluator.load_results(args.results_path, dataset_filter=args.dataset_filter)
        evaluated_results, stats = evaluator.evaluate_dataset(results, start_idx)
        
        evaluator.save_results(evaluated_results, stats, args.output_path)
        
        if os.path.exists(interrupted_file):
            os.remove(interrupted_file)
        
        evaluator.print_summary(stats)
        
        print("\nEvaluation completed!")
        return 0
    
    except Exception as e:
        print(f"ERROR: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
