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
"""
Implement Actor
"""

import os
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
import numpy as np
from einops import rearrange
from ray.experimental.tqdm_ray import tqdm
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.nn.functional as F

from ...protocol import DataProto, batch_collate
from ...trainer.core_algos import average_loss, compute_kl, compute_policy_loss
from ...utils import torch_functional as VF
from ...utils.py_functional import append_to_dict
from ...utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from ...utils.ulysses import gather_outputs_and_unpad, ulysses_pad_and_slice_inputs
from .base import BasePPOActor
from .config import ActorConfig


try:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
except ImportError:
    pass


__all__ = ["DataParallelPPOActor"]

def anneal_branch_sampler(micro_batch: dict[str, torch.Tensor], anneal_prob: float = 0.0) -> dict[str, torch.Tensor]:
    """
    Anneal-based branch sampler for micro_batch dictionaries.

    Each micro_batch has keys like 'input_ids_A', 'input_ids_B', etc.
    With probability `anneal_prob`, select A branch; otherwise B branch.
    Returns a new dictionary with unified keys (no _A/_B suffixes).
    """

    # --- Step 1. Prepare probability mask ---
    device = next(iter(micro_batch.values())).device
    batch_size = next(iter(micro_batch.values())).size(0)
    branch_mask = torch.bernoulli(
        torch.full((batch_size,), float(anneal_prob), device=device, dtype=torch.float32)
    ).bool()

    merged = {}

    # --- Step 2. Merge tensor branches ---
    keys_A = [k for k in micro_batch.keys() if k.endswith("_A")]
    for key_A in keys_A:
        key_base = key_A[:-2]
        key_B = key_base + "_B"
        if key_B not in micro_batch:
            raise KeyError(f"Missing paired key for {key_A}")

        tensor_A = micro_batch[key_A]
        tensor_B = micro_batch[key_B]

        if tensor_A.shape != tensor_B.shape:
            raise ValueError(f"Shape mismatch: {key_A} {tensor_A.shape} vs {key_B} {tensor_B.shape}")

        mask_exp = branch_mask.view(-1, *([1] * (tensor_A.ndim - 1)))
        merged[key_base] = torch.where(mask_exp, tensor_A, tensor_B)

    # --- Step 3. Keep keys without _A/_B suffix (meta / non-dual fields) ---
    for key, val in micro_batch.items():
        if not (key.endswith("_A") or key.endswith("_B")):
            merged[key] = val

    # --- Step 4. Return merged micro_batch ---
    merged["branch_mask"] = branch_mask  # Optional: for debug / metrics
    return merged


class DataParallelPPOActor(BasePPOActor):
    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        When optimizer is None, it is Reference Policy
        """
        super().__init__(config)
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.dual_mode = False
        if config.use_torch_compile:
            self.log_probs_from_logits = torch.compile(VF.log_probs_from_logits, dynamic=True)
        else:
            self.log_probs_from_logits = VF.log_probs_from_logits

    def _forward_micro_batch(self, micro_batch: dict[str, torch.Tensor], temperature: float, dual_mode: bool = False) -> torch.Tensor:
        """
        Returns:
            log_probs: # (bs, response_len)
        """
        input_ids = micro_batch["input_ids"]
        batch_size, seqlen = input_ids.shape
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

        multi_modal_inputs = defaultdict(list)
        if "multi_modal_inputs" in micro_batch:
            multi_modal_inputs = batch_collate(micro_batch["multi_modal_inputs"])
            multi_modal_inputs = {key: torch.cat(value, dim=0) for key, value in multi_modal_inputs.items()}
        else:
            multi_modal_inputs = {}

        if self.config.padding_free:
            input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # (total_nnz, 1)
            input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

            # unpad the position_ids to align the rotary
            if position_ids.dim() == 3:
                position_ids_rmpad = (
                    index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                    .transpose(0, 1)
                    .unsqueeze(1)
                )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
            else:
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

            # for compute the log_prob
            input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

            # pad and slice the inputs if sp > 1
            if self.config.ulysses_size > 1:
                input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=self.config.ulysses_size
                )
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, self.config.ulysses_size
                )

            input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

            # only pass input_ids and position_ids to enable flash_attn_varlen
            output = self.actor_module(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids_rmpad,
                **multi_modal_inputs,
                use_cache=False,
            )  # prevent model thinks we are generating
            logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
            logits_rmpad.div_(temperature)
            # ((total_nnz / sp) + pad)
            log_probs = self.log_probs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

            # gather log_prob if sp > 1
            if self.config.ulysses_size > 1:
                # gather and unpad for the ulysses sp
                log_probs = gather_outputs_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)

            # pad back to (bsz, seqlen)
            full_log_probs = pad_input(
                hidden_states=log_probs.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
            )
            log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
        else:
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **multi_modal_inputs,
                use_cache=False,
            )
            logits: torch.Tensor = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            log_probs = self.log_probs_from_logits(logits, responses)  # (bsz, response_length)

        if not dual_mode:
            return log_probs
        else:
            if self.config.padding_free:
                log_probs_full_vocab = F.log_softmax(logits_rmpad, dim=-1)
                if self.config.ulysses_size > 1:
                    log_probs_full_vocab = gather_outputs_and_unpad(
                        log_probs_full_vocab, gather_dim=0, unpad_dim=0, padding_size=pad_size
                    )
                log_probs_full_vocab = pad_input(
                    hidden_states=log_probs_full_vocab, indices=indices, batch=batch_size, seqlen=seqlen
                )
                log_probs_full_vocab = log_probs_full_vocab[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
            else:
                log_probs_full_vocab = F.log_softmax(logits, dim=-1)

            return log_probs, log_probs_full_vocab.detach()

    
    def _compute_visual_uncertainty(
        self,
        cross_log_probs: torch.Tensor,  # [B, seq, vocab]
        log_probs: torch.Tensor,        # [B, seq, vocab]
        response_mask: torch.Tensor,    # [B, seq]
        mode: str = "token_level",      # or "sequence_level"
    ) -> torch.Tensor:
        kl_pq = F.kl_div(cross_log_probs, log_probs, reduction="none", log_target=True)
        kl_qp = F.kl_div(log_probs, cross_log_probs, reduction="none", log_target=True)
        sym_kl = 0.5 * (kl_pq + kl_qp)  # [B, seq, vocab]

        sym_kl = sym_kl.sum(dim=-1)  # [B, seq]

        sym_kl = sym_kl * response_mask  # [B, seq]

        if mode == "sequence_level":
            valid_count = response_mask.sum(dim=-1).clamp_min(1)
            return sym_kl.sum(dim=-1) / valid_count  # [B]
        elif mode == "token_level":
            return sym_kl  # [B, seq]
        else:
            raise ValueError(f"Unknown mode {mode} for visual uncertainty computation.")

    def _forward_micro_batch_cross_probs(self, model_inputs: dict[str, torch.Tensor], temperature: float) -> torch.Tensor:
        """
        Compute cross log-probs: (B prompt + A response).
        `model_inputs` is a flat dict containing *_A and *_B tensors.
        """
        if not any(k.endswith("_A") for k in model_inputs.keys()):
            raise ValueError("cross probs expects dual-branch inputs with *_A/_B keys")

        # split A/B views
        batch_A = {k[:-2]: v for k, v in model_inputs.items() if k.endswith("_A")}
        batch_B = {k[:-2]: v for k, v in model_inputs.items() if k.endswith("_B")}

        resp_len_B = batch_B["responses"].size(1)

        cross_batch = {
            # take B prompt (all tokens except its response tail) + A response
            "input_ids":       torch.cat([batch_B["input_ids"][:, :-resp_len_B], batch_A["responses"]], dim=-1),
            "attention_mask":  torch.cat([batch_B["attention_mask"][:, :-resp_len_B], batch_A["response_mask"]], dim=-1),
            "position_ids":    batch_B["position_ids"],
            "responses":       batch_A["responses"],
            "response_mask":   batch_A["response_mask"],
        }
        if "multi_modal_inputs_B" in model_inputs:
            cross_batch["multi_modal_inputs"] = model_inputs["multi_modal_inputs_B"]

        return self._forward_micro_batch(cross_batch, temperature=temperature, dual_mode=self.dual_mode)

    def _optimizer_step(self) -> torch.Tensor:
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(self.config.max_grad_norm)
        else:
            grad_norm = nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.max_grad_norm)

        if not torch.isfinite(grad_norm):
            print("Gradient norm is not finite. Skip update.")
        else:
            self.actor_optimizer.step()

        self.actor_optimizer.zero_grad()
        return grad_norm

    @torch.no_grad()
    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        self.actor_module.eval()

        temperature = data.meta_info["temperature"]
        select_keys = ["input_ids", "attention_mask", "position_ids", "responses"]
        non_tensor_select_keys = ["multi_modal_inputs"]

        data = data.select(select_keys, non_tensor_select_keys)
        if self.config.dynamic_batching:
            max_token_len = self.config.micro_batch_size_per_device_for_experience * data.batch["input_ids"].size(-1)
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(self.config.micro_batch_size_per_device_for_experience)

        log_probs_lst = []
        if self.rank == 0:
            micro_batches = tqdm(micro_batches, desc="Compute log probs", position=1)

        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            try:
                log_probs = self._forward_micro_batch(model_inputs, temperature=temperature)
            except Exception as e:
                print(f"Error in _forward_micro_batch: {e}")
                breakpoint()
                raise e
            log_probs_lst.append(log_probs)

        log_probs = torch.concat(log_probs_lst, dim=0)

        if self.config.dynamic_batching:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)

        return log_probs
        
    def update_policy(self, data: DataProto) -> dict[str, Any]:
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid slient error
        base_keys = ["input_ids", "attention_mask", "position_ids", "responses", "response_mask"]
        extra_keys = ["old_log_probs", "ref_log_probs", "advantages"]
        non_tensor_select_keys = ["multi_modal_inputs"]
        global_step = data.meta_info.get("global_step", None)
        total_steps = data.meta_info.get("total_steps", None)

        # detect dual-branch case
        self.dual_mode = any(k.endswith("_A") for k in data.batch.keys())

        if self.dual_mode:
            # add both A and B branch versions
            select_keys = [f"{k}_A" for k in base_keys + extra_keys] + [f"{k}_B" for k in base_keys + extra_keys]
            non_tensor_select_keys = ["multi_modal_inputs_A", "multi_modal_inputs_B"]
        else:
            select_keys = base_keys + extra_keys
            non_tensor_select_keys = ["multi_modal_inputs"]

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.select(select_keys, non_tensor_select_keys).split(self.config.global_batch_size_per_device)

        metrics = defaultdict(list)
        for _ in range(self.config.ppo_epochs):
            if self.rank == 0:
                mini_batches = tqdm(mini_batches, desc="Train mini-batches", position=1)

            for mini_batch in mini_batches:
                if self.config.dynamic_batching and not self.dual_mode:
                    max_input_len = mini_batch.batch["input_ids"].size(-1)
                    max_token_len = self.config.micro_batch_size_per_device_for_update * max_input_len
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.micro_batch_size_per_device_for_update)

                if self.rank == 0:
                    micro_batches = tqdm(micro_batches, desc="Update policy", position=2)

                for micro_batch in micro_batches:
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

                    if not self.dual_mode:
                        total_response_tokens = torch.sum(mini_batch.batch["response_mask"])
                        dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)
                        response_mask = model_inputs["response_mask"]
                        old_log_probs = model_inputs["old_log_probs"]
                        advantages = model_inputs["advantages"]

                        log_probs = self._forward_micro_batch(model_inputs, temperature=temperature, dual_mode=self.dual_mode)

                    else:
                        # === dual-branch ===
                        # branch A
                        model_inputs_A = {k[:-2]: v for k, v in model_inputs.items() if k.endswith("_A")}
                        response_mask_A = model_inputs_A["response_mask"]
                        old_log_probs_A = model_inputs_A["old_log_probs"]
                        advantages_A = model_inputs_A["advantages"]

                        log_probs_A, log_probs_A_full_vocab = self._forward_micro_batch(model_inputs_A, temperature=temperature, dual_mode=self.dual_mode)

                        # branch B
                        model_inputs_B = {k[:-2]: v for k, v in model_inputs.items() if k.endswith("_B")}
                        response_mask_B = model_inputs_B["response_mask"]
                        old_log_probs_B = model_inputs_B["old_log_probs"]
                        advantages_B = model_inputs_B["advantages"]

                        log_probs_B, _ = self._forward_micro_batch(model_inputs_B, temperature=temperature, dual_mode=self.dual_mode)
                        _, cross_log_probs_full_vocab = self._forward_micro_batch_cross_probs(model_inputs, temperature=temperature)
                        
                        # compute visual uncertainty and adjust advantages_B
                        visual_uncertainty = self._compute_visual_uncertainty(cross_log_probs_full_vocab, log_probs_A_full_vocab, response_mask_A, mode="token_level")
                        
                        del cross_log_probs_full_vocab, log_probs_A_full_vocab
                        torch.cuda.empty_cache()
                        
                        # advantages_B = advantages_B + torch.min(
                        #     advantages_B.abs(),
                        #     self.config.visual_uncertainty_coef * visual_uncertainty
                        # )
                        anneal_prob = 1.0
                        device = log_probs_A.device
                        batch_size = log_probs_A.size(0)
                        # breakpoint()
                        branch_mask = torch.bernoulli(torch.full((batch_size,), anneal_prob, device=device)).bool()

                        mask_exp = branch_mask.view(-1, 1)

                        log_probs     = torch.where(mask_exp, log_probs_A, log_probs_B)
                        old_log_probs = torch.where(mask_exp, old_log_probs_A, old_log_probs_B)
                        advantages    = torch.where(mask_exp, advantages_A, advantages_B)
                        response_mask = torch.where(mask_exp, response_mask_A, response_mask_B)
                        total_response_tokens = torch.sum(response_mask)
                        dist.all_reduce(total_response_tokens, op=dist.ReduceOp.SUM)

                    pg_loss, pg_metrics = compute_policy_loss(
                        old_log_probs=old_log_probs,
                        log_probs=log_probs,
                        advantages=advantages,
                        response_mask=response_mask,
                        clip_ratio_low=self.config.clip_ratio_low,
                        clip_ratio_high=self.config.clip_ratio_high,
                        clip_ratio_dual=self.config.clip_ratio_dual,
                        loss_avg_mode=self.config.loss_avg_mode,
                    )
                    if self.config.use_kl_loss and "ref_log_probs" in model_inputs:
                        ref_log_probs = model_inputs["ref_log_probs"]
                        # compute kl loss
                        kld = compute_kl(
                            log_probs=log_probs,
                            ref_log_probs=ref_log_probs,
                            kl_penalty=self.config.kl_penalty,
                        )
                        kl_loss = average_loss(kld, response_mask, mode=self.config.loss_avg_mode)
                        loss = pg_loss + kl_loss * self.config.kl_coef
                        metrics["actor/kl_loss"] = kl_loss.detach().item()
                        metrics["actor/kl_coef"] = self.config.kl_coef
                    else:
                        loss = pg_loss

                    loss = loss * torch.sum(response_mask) * self.world_size / total_response_tokens
                    loss.backward()

                    batch_metrics = {
                        "actor/pg_loss": pg_loss.detach().item(),
                        "actor/pg_clipfrac_higher": pg_metrics["pg_clipfrac_higher"],
                        "actor/pg_clipfrac_lower": pg_metrics["pg_clipfrac_lower"],
                        "actor/entropy_loss": pg_metrics["entropy_loss"],
                        "actor/ppo_kl": pg_metrics["ppo_kl"],
                    }
                    append_to_dict(metrics, batch_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        return metrics
