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

from typing import Any
try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(reward_inputs: list[dict[str, Any]], timeout_score: float = 0) -> bool:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for math reward function.")

    scores = []
    for reward_input in reward_inputs:
        model_output = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        verify_func = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )
        ret_score = 0.0

        # Wrap the ground truth in \boxed{} format for verification
        ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        try:
            ret_score, _ = verify_func([ground_truth_boxed], [model_output])
        except Exception:
            pass
        except TimeoutException:
            ret_score = timeout_score
        scores.append(
            {
                "overall": ret_score,
            }
        )

    return scores