# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Any
from rapidfuzz.distance import Levenshtein
from mathruler.grader import extract_boxed_content, grade_answer


def format_reward(response: str) -> float:
    pattern = re.compile(r"<think>.*</think>.*\\boxed\{.*\}.*", re.DOTALL)
    format_match = re.fullmatch(pattern, response)
    return 1.0 if format_match else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    """Reward for exact match between predicted and ground-truth boxed answers."""
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def edit_distance_reward(response: str, ground_truth: str) -> float:
    """Reward based on normalized Levenshtein similarity between response and ground-truth."""
    ans = extract_boxed_content(response)
    if not ans or not ground_truth:
        return 0.0
    return Levenshtein.normalized_similarity(ans.strip(), ground_truth.strip())


def has_numeric_answer(text: str) -> bool:
    """Check whether the text contains numeric patterns (integer or float)."""
    if not text:
        return False
    return bool(re.search(r"-?\d+(?:\.\d+)?", text))


def numeric_tolerance_reward(response: str, ground_truth: str, tol: float = 0.05) -> float:
    """
    Reward based on numeric tolerance.
    - Full score if relative error <= tol.
    - Linear decay beyond the tolerance, down to 0.
    """
    extract = lambda s: re.findall(r"-?\d+(?:\.\d+)?", s)
    pred_nums = extract(extract_boxed_content(response))
    gt_nums = extract(ground_truth)
    if not pred_nums or not gt_nums:
        return 0.0

    try:
        pred_val = float(pred_nums[-1])  # last number usually represents the final answer
        gt_val = float(gt_nums[-1])

        if abs(gt_val) < 1e-8:
            return 1.0 if abs(pred_val - gt_val) < 1e-8 else 0.0

        rel_err = abs(pred_val - gt_val) / abs(gt_val)
        if rel_err <= tol:
            return 1.0
        else:
            # beyond tolerance, apply linear decay
            return max(0.0, 1 - (rel_err - tol) * 5)
    except ValueError:
        return 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    format_weight: float = 0.1,
    problem_recog_weight: float = 0.05,
) -> list[dict[str, float]]:
    """
    Reward computation logic:
      - Geometry3K dataset: strict accuracy only.
      - Other datasets:
          * If ground-truth contains numbers → numeric_tolerance_reward (±5%)
          * Otherwise → edit_distance_reward (semantic similarity)
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])  # normalize formatting
        dataset = reward_input.get("dataset", "")
        ground_truth = reward_input["ground_truth"]

        format_score = format_reward(response)

        if dataset == "Geometry3K":
            # strict match (no tolerance)
            accuracy_score = accuracy_reward(response, ground_truth)
        else:
            # for non-Geometry datasets
            if has_numeric_answer(ground_truth):
                # apply numeric tolerance ±5%
                accuracy_score = numeric_tolerance_reward(response, ground_truth, tol=0.05)
            else:
                # use edit distance for non-numeric responses
                accuracy_score = edit_distance_reward(response, ground_truth)

        overall = (1 - format_weight) * accuracy_score + format_weight * format_score
        scores.append(
            {
                "overall": overall,
                "format": format_score,
                "accuracy": accuracy_score,
            }
        )

    return scores
