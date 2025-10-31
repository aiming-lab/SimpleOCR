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

import numpy as np
import torch

from ..protocol import DataProto


def reduce_metrics(metrics: dict[str, list[Any]]) -> dict[str, Any]:
    return {key: np.mean(value) for key, value in metrics.items()}


# def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
#     max_response_length = batch.batch["responses"].size(-1)
#     max_prompt_length = batch.batch["attention_mask"].size(-1) - max_response_length

#     prompt_length = batch.batch["attention_mask"][:, :-max_response_length].sum(-1).float()
#     response_length = batch.batch["attention_mask"][:, -max_response_length:].sum(-1).float()

#     return {
#         # response length
#         "response_length/mean": torch.mean(response_length).detach().item(),
#         "response_length/max": torch.max(response_length).detach().item(),
#         "response_length/min": torch.min(response_length).detach().item(),
#         "response_length/clip_ratio": torch.eq(response_length, max_response_length).float().mean().detach().item(),
#         # prompt length
#         "prompt_length/mean": torch.mean(prompt_length).detach().item(),
#         "prompt_length/max": torch.max(prompt_length).detach().item(),
#         "prompt_length/min": torch.min(prompt_length).detach().item(),
#         "prompt_length/clip_ratio": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
#     }

def compute_length_metrics(batch: DataProto) -> dict[str, Any]:
    """
    Compute length statistics (prompt/response length).
    Compatible with both single-branch and dual-branch batches.
    """

    b = batch.batch
    metrics = {}

    def compute_for_branch(suffix: str = ""):
        responses = b[f"responses{suffix}"]
        attn_mask = b[f"attention_mask{suffix}"]

        max_response_length = responses.size(-1)
        max_prompt_length = attn_mask.size(-1) - max_response_length

        prompt_length = attn_mask[:, :-max_response_length].sum(-1).float()
        response_length = attn_mask[:, -max_response_length:].sum(-1).float()

        s = f"_{suffix[1:]}" if suffix else ""
        return {
            f"response_length/mean{s}": torch.mean(response_length).detach().item(),
            f"response_length/max{s}": torch.max(response_length).detach().item(),
            f"response_length/min{s}": torch.min(response_length).detach().item(),
            f"response_length/clip_ratio{s}": torch.eq(response_length, max_response_length).float().mean().detach().item(),
            f"prompt_length/mean{s}": torch.mean(prompt_length).detach().item(),
            f"prompt_length/max{s}": torch.max(prompt_length).detach().item(),
            f"prompt_length/min{s}": torch.min(prompt_length).detach().item(),
            f"prompt_length/clip_ratio{s}": torch.eq(prompt_length, max_prompt_length).float().mean().detach().item(),
        }

    # --- single or dual branch ---
    if any(k.endswith("_A") for k in b.keys()):
        metrics.update(compute_for_branch("_A"))
        metrics.update(compute_for_branch("_B"))
    else:
        metrics.update(compute_for_branch(""))

    return metrics

# def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
#     sequence_score = batch.batch["token_level_scores"].sum(-1)
#     sequence_reward = batch.batch["token_level_rewards"].sum(-1)

#     advantages = batch.batch["advantages"]
#     returns = batch.batch["returns"]

#     max_response_length = batch.batch["responses"].size(-1)
#     response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

#     valid_adv = torch.masked_select(advantages, response_mask)
#     valid_returns = torch.masked_select(returns, response_mask)

#     if use_critic:
#         values = batch.batch["values"]
#         valid_values = torch.masked_select(values, response_mask)
#         return_diff_var = torch.var(valid_returns - valid_values)
#         return_var = torch.var(valid_returns)

#     return {
#         # score
#         "critic/score/mean": torch.mean(sequence_score).detach().item(),
#         "critic/score/max": torch.max(sequence_score).detach().item(),
#         "critic/score/min": torch.min(sequence_score).detach().item(),
#         # reward
#         "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
#         "critic/rewards/max": torch.max(sequence_reward).detach().item(),
#         "critic/rewards/min": torch.min(sequence_reward).detach().item(),
#         # adv
#         "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
#         "critic/advantages/max": torch.max(valid_adv).detach().item(),
#         "critic/advantages/min": torch.min(valid_adv).detach().item(),
#         # returns
#         "critic/returns/mean": torch.mean(valid_returns).detach().item(),
#         "critic/returns/max": torch.max(valid_returns).detach().item(),
#         "critic/returns/min": torch.min(valid_returns).detach().item(),
#         **(
#             {
#                 # values
#                 "critic/values/mean": torch.mean(valid_values).detach().item(),
#                 "critic/values/max": torch.max(valid_values).detach().item(),
#                 "critic/values/min": torch.min(valid_values).detach().item(),
#                 # vf explained var
#                 "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
#             }
#             if use_critic
#             else {}
#         ),
#         **compute_length_metrics(batch),
#     }

def compute_data_metrics(batch: DataProto, use_critic: bool = False) -> dict[str, Any]:
    """
    Compute reward/score/advantage metrics.
    Compatible with both single-branch and dual-branch batches.
    """

    b = batch.batch
    metrics = {}

    def compute_for_branch(prefix: str = "", suffix: str = ""):
        scores = b[f"token_level_scores{suffix}"]
        rewards = b[f"token_level_rewards{suffix}"]
        advantages = b[f"advantages{suffix}"]
        returns = b[f"returns{suffix}"]

        responses = b[f"responses{suffix}"]
        attn_mask = b[f"attention_mask{suffix}"]

        max_response_length = responses.size(-1)
        response_mask = attn_mask[:, -max_response_length:].bool()

        valid_adv = torch.masked_select(advantages, response_mask)
        valid_returns = torch.masked_select(returns, response_mask)

        if use_critic and f"values{suffix}" in b:
            values = b[f"values{suffix}"]
            valid_values = torch.masked_select(values, response_mask)
            return_diff_var = torch.var(valid_returns - valid_values)
            return_var = torch.var(valid_returns)
        else:
            valid_values = None
            return_diff_var = return_var = None

        result = {
            f"{prefix}/score/mean": torch.mean(scores.sum(-1)).item(),
            f"{prefix}/score/max": torch.max(scores.sum(-1)).item(),
            f"{prefix}/score/min": torch.min(scores.sum(-1)).item(),
            f"{prefix}/rewards/mean": torch.mean(rewards.sum(-1)).item(),
            f"{prefix}/rewards/max": torch.max(rewards.sum(-1)).item(),
            f"{prefix}/rewards/min": torch.min(rewards.sum(-1)).item(),
            f"{prefix}/advantages/mean": torch.mean(valid_adv).item(),
            f"{prefix}/advantages/max": torch.max(valid_adv).item(),
            f"{prefix}/advantages/min": torch.min(valid_adv).item(),
            f"{prefix}/returns/mean": torch.mean(valid_returns).item(),
            f"{prefix}/returns/max": torch.max(valid_returns).item(),
            f"{prefix}/returns/min": torch.min(valid_returns).item(),
        }

        if use_critic and valid_values is not None:
            result.update({
                f"{prefix}/values/mean": torch.mean(valid_values).item(),
                f"{prefix}/values/max": torch.max(valid_values).item(),
                f"{prefix}/values/min": torch.min(valid_values).item(),
                f"{prefix}/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).item(),
            })

        return result

    # --- single or dual branch ---
    if any(k.endswith("_A") for k in b.keys()):
        metrics.update(compute_for_branch("critic_A", "_A"))
        metrics.update(compute_for_branch("critic_B", "_B"))
    else:
        metrics.update(compute_for_branch("critic", ""))

    # --- add length metrics ---
    metrics.update(compute_length_metrics(batch))
    return metrics

# def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
#     num_response_tokens = torch.sum(batch.batch["response_mask"]).item()
#     num_overall_tokens = sum(batch.meta_info["global_token_num"])
#     num_tokens_of_section = {
#         **dict.fromkeys(["gen", "reward"], num_response_tokens),
#         **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
#     }
#     return {
#         **{f"timing_s/{name}": value for name, value in timing_raw.items()},
#         **{
#             f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name]
#             for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
#         },
#     }

def compute_timing_metrics(batch: DataProto, timing_raw: dict[str, float]) -> dict[str, Any]:
    """
    Compute timing and per-token speed.
    Supports dual-branch (A/B) structure.
    """
    b = batch.batch
    metrics = {}

    def compute_for_branch(suffix: str = "", tag: str = ""):
        num_response_tokens = torch.sum(b[f"response_mask{suffix}"]).item()
        num_overall_tokens = sum(batch.meta_info["global_token_num"])

        num_tokens_of_section = {
            **dict.fromkeys(["gen", "reward"], num_response_tokens),
            **dict.fromkeys(["ref", "old", "values", "adv", "update_critic", "update_actor"], num_overall_tokens),
        }

        result = {
            **{f"timing_s/{name}{tag}": value for name, value in timing_raw.items()},
            **{
                f"timing_per_token_ms/{name}{tag}": timing_raw[name] * 1000 / num_tokens_of_section[name]
                for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())
            },
        }
        return result

    if any(k.endswith("_A") for k in b.keys()):
        metrics.update(compute_for_branch("_A", "_A"))
        metrics.update(compute_for_branch("_B", "_B"))
    else:
        metrics.update(compute_for_branch("", ""))

    return metrics

# def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
#     total_num_tokens = sum(batch.meta_info["global_token_num"])
#     time = timing_raw["step"]
#     return {
#         "perf/total_num_tokens": total_num_tokens,
#         "perf/time_per_step": time,
#         "perf/throughput": total_num_tokens / (time * num_gpus),
#     }

def compute_throughout_metrics(batch: DataProto, timing_raw: dict[str, float], num_gpus: int) -> dict[str, Any]:
    """
    Compute throughput and total tokens per step.
    Dual-branch compatible.
    """

    def compute_for_branch(meta_info, tag: str = ""):
        total_num_tokens = sum(meta_info["global_token_num"])
        time = timing_raw["step"]
        return {
            f"perf/total_num_tokens{tag}": total_num_tokens,
            f"perf/time_per_step{tag}": time,
            f"perf/throughput{tag}": total_num_tokens / (time * num_gpus),
        }

    b = batch.batch
    metrics = {}

    if any(k.endswith("_A") for k in b.keys()):
        metrics.update(compute_for_branch(batch.meta_info, "_A"))
        metrics.update(compute_for_branch(batch.meta_info, "_B"))
    else:
        metrics.update(compute_for_branch(batch.meta_info, ""))

    return metrics