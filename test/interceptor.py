import torch
import torch.nn.functional as F
from typing import Optional, Callable, List, Dict, Any, Tuple

# --- Local RoPE helpers (match Qwen3's behavior) ---

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the features (standard RoPE helper)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE with cos/sin shaped [B, T, H] to q/k shaped [B, heads, T, H].
    We unsqueeze on head dimension for broadcasting (dim=1).
    """
    cos = cos.unsqueeze(1)  # [B, 1, T, H]
    sin = sin.unsqueeze(1)  # [B, 1, T, H]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class QKVInterceptorQwen3Eager:
    """
    Intercepts Q/K/V inside Qwen3Attention while preserving the official, optimized
    `eager_attention_forward`. Two intervention points are supported:

      (2) Post-norm, pre-RoPE:    after q_norm / k_norm, before RoPE is applied.
      (3) Post-RoPE, pre-scores:  after RoPE, right before attention score computation.

    Key properties:
      - Fully respects the forward signature of Qwen3Attention and returns the same outputs.
      - Works with GQA: we intercept K/V BEFORE the internal repeat (done inside eager_attention_forward).
      - Preserves caching, masks, sliding window, dropout, and o_proj.
      - Clean context-manager API and optional tensor capture for analysis.

    Limitations:
      - Designed for the "eager" attention path. Other backends will still pass through,
        but the interceptor is intended for eager to keep behavior predictable.
    """

    def __init__(
        self,
        model,
        on_stage2: Optional[Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        on_stage3: Optional[Callable[[int, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = None,
        save_tensors: bool = True,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            model:             A Qwen3 model from HF Transformers (e.g., Qwen3ForCausalLM.model).
            on_stage2:         Callback applied AFTER q_norm/k_norm, BEFORE RoPE.
                               Signature: (layer_idx, q, k, v) -> (q, k, v), with shapes [B, heads, T, H].
            on_stage3:         Callback applied AFTER RoPE, BEFORE attention scores (same signature).
            save_tensors:      If True, snapshots are pushed to `self.captures` for later analysis (on CPU).
            verbose:           If True, prints which attention layers are patched.
        """
        self.model = model
        self.on_stage2 = on_stage2
        self.on_stage3 = on_stage3
        self.save = save_tensors
        self.verbose = verbose

        self._patches: List[Tuple[torch.nn.Module, Callable]] = []
        self.captures: List[Dict[str, Any]] = []

    def __enter__(self):
        layer_idx = 0
        for name, module in self.model.named_modules():
            # Match the concrete class name directly to avoid version-specific imports
            if module.__class__.__name__ == "Qwen3Attention":
                if self.verbose:
                    print(f"[QKVInterceptor] Hooking Qwen3Attention at layer {layer_idx}: {name}")

                original_forward = module.forward

                def make_wrapper(idx: int, attn: torch.nn.Module, orig_fwd: Callable):
                    """
                    Wrapped forward that mirrors Qwen3Attention.forward while inserting:
                      - Stage 2 hook: post-norm/pre-RoPE
                      - Stage 3 hook: post-RoPE/pre-scores
                    All other logic remains identical, including:
                      - head shaping
                      - repeat_kv (performed by eager_attention_forward)
                      - cache update
                      - masking / sliding window
                      - dropout semantics
                      - o_proj and return signature
                    """
                    def wrapped_forward(
                        hidden_states: torch.Tensor,
                        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                        attention_mask: Optional[torch.Tensor],
                        past_key_values: Optional["Cache"] = None,
                        cache_position: Optional[torch.LongTensor] = None,
                        **kwargs,
                    ):
                        # === Shapes & params (match official code) ===
                        input_shape = hidden_states.shape[:-1]         # (B, T)
                        head_dim   = getattr(attn, "head_dim")
                        # Target per-head view: (*input_shape, -1, head_dim) then transpose to [B, heads, T, H]
                        hidden_shape = (*input_shape, -1, head_dim)

                        # === Projections + per-head reshape ===
                        q = attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, h_q, T, H]
                        k = attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, h_kv, T, H]
                        v = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # [B, h_kv, T, H]

                        # === Qwen3-specific per-head RMSNorm (after reshape) ===
                        q = attn.q_norm(q)
                        k = attn.k_norm(k)

                        # === Stage 2: post-norm, pre-RoPE ===
                        if self.save:
                            self.captures.append({
                                "stage": "post_norm_pre_rope",
                                "layer_idx": idx,
                                "q": q.detach().cpu(),
                                "k": k.detach().cpu(),
                                "v": v.detach().cpu(),
                            })
                        if self.on_stage2 is not None:
                            q, k, v = self.on_stage2(idx, q, k, v)

                        # === Apply RoPE using the already-prepared shared position embeddings ===
                        cos, sin = position_embeddings  # [B, T, H]
                        q, k = _apply_rope(q, k, cos, sin)

                        # === Stage 3: post-RoPE, pre-scores ===
                        if self.save:
                            self.captures.append({
                                "stage": "post_rope_pre_scores",
                                "layer_idx": idx,
                                "q": q.detach().cpu(),
                                "k": k.detach().cpu(),
                                "v": v.detach().cpu(),
                            })
                        if self.on_stage3 is not None:
                            q, k, v = self.on_stage3(idx, q, k, v)

                        # === Cache update (unchanged) ===
                        if past_key_values is not None:
                            # RoPE-specific cache kwargs
                            cos_, sin_ = cos, sin
                            cache_kwargs = {"sin": sin_, "cos": cos_, "cache_position": cache_position}
                            k, v = past_key_values.update(k, v, attn.layer_idx, cache_kwargs)

                        # === Choose attention backend (keep default behavior) ===
                        # Default is eager_attention_forward; otherwise backend from config is used.
                        attention_interface = F  # placeholder to avoid lints; will be overwritten
                        if getattr(attn.config, "_attn_implementation", "eager") == "eager":
                            from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward as _eager
                            attention_interface = _eager
                        else:
                            from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
                            attention_interface = ALL_ATTENTION_FUNCTIONS[attn.config._attn_implementation]

                        # === Call the official attention interface (handles repeat_kv, masks, sliding window, etc.) ===
                        attn_output, attn_weights = attention_interface(
                            attn,
                            q,           # [B, h_q, T, H]
                            k,           # [B, h_kv, T, H] (repeat_kv happens inside interface)
                            v,           # [B, h_kv, T, H]
                            attention_mask,
                            dropout=0.0 if not attn.training else attn.attention_dropout,
                            scaling=attn.scaling,
                            sliding_window=getattr(attn, "sliding_window", None),
                            **kwargs,
                        )

                        # === Output projection and return ===
                        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                        attn_output = attn.o_proj(attn_output)

                        # Match Qwen3Attention return type (attn_weights optional)
                        need_weights = kwargs.get("output_attentions", False)
                        if need_weights:
                            return attn_output, attn_weights
                        return attn_output, None

                    return wrapped_forward

                module.forward = make_wrapper(layer_idx, module, original_forward)
                self._patches.append((module, original_forward))
                layer_idx += 1

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Restore original forwards
        for module, orig in self._patches:
            module.forward = orig
        if self.verbose:
            print("[QKVInterceptor] All Qwen3Attention hooks have been removed.")
