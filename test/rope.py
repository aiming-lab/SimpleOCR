import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def build_rope_cache(
    seq_len: int,
    dim: int,
    base: float = 10000.0,
    device=None,
    dtype=torch.float32,
    rope_dim: int | None = None,
):
    """
    Build cosine and sine caches used in Rotary Positional Embedding (RoPE).

    Args:
        seq_len (int): Maximum sequence length.
        dim (int): Total hidden dimension (usually head_dim for attention heads).
        base (float): Frequency base (commonly 10000.0).
        device: Torch device for returned tensors.
        dtype: Torch dtype for returned tensors.
        rope_dim (int | None): Number of dimensions to apply RoPE on.
                               If None, defaults to the full dimension.

    Returns:
        cos (Tensor): Shape [seq_len, rope_dim // 2].
        sin (Tensor): Shape [seq_len, rope_dim // 2].
        rope_dim (int): Actual dimensionality used for rotation.
    """
    if rope_dim is None:
        rope_dim = dim
    assert rope_dim % 2 == 0, "rope_dim must be even for rotation pairs."

    device = device or torch.device("cpu")

    # Compute inverse frequencies for each rotation pair
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, device=device, dtype=dtype) / rope_dim)
    )  # [rope_dim // 2]

    # Position indices [0, 1, 2, ..., seq_len-1]
    t = torch.arange(seq_len, device=device, dtype=dtype)

    # Outer product to get all rotation angles: [seq_len, rope_dim // 2]
    freqs = torch.einsum("n,f->nf", t, inv_freq)

    # Precompute cosine and sine values
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    return cos, sin, rope_dim


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    """
    Apply Rotary Positional Embedding to input tensor (typically Q or K).

    Args:
        x (Tensor): Input tensor of shape [..., seq_len, dim].
        cos (Tensor): Cached cos values, shape [seq_len, rope_dim // 2].
        sin (Tensor): Cached sin values, shape [seq_len, rope_dim // 2].
        rope_dim (int): Dimensionality to apply rotation on.

    Returns:
        Tensor: Tensor with RoPE applied, same shape as input.
    """
    x_rope, x_rest = x[..., :rope_dim], x[..., rope_dim:]

    # Reshape to [..., seq_len, rope_dim/2, 2] for paired dimensions
    x_rope = x_rope.view(*x_rope.shape[:-1], rope_dim // 2, 2)
    x1, x2 = x_rope[..., 0], x_rope[..., 1]  # split each pair dim

    # Expand cos/sin to match x dims
    while cos.dim() < x1.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # Rotation:
    # x1' = x1*cos - x2*sin
    # x2' = x1*sin + x2*cos
    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos

    x_rot = torch.stack([x1_new, x2_new], dim=-1).reshape(*x_rope.shape[:-2], rope_dim)
    return torch.cat([x_rot, x_rest], dim=-1)


def compute_similarity_curve(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    """
    Compute cosine similarity between position 0 and position d (0 to seq_len-1)
    after applying RoPE.

    Args:
        x (Tensor): shape [L, D], no rotation yet.
        cos, sin, rope_dim: RoPE parameters.
    Returns:
        List of similarities of length L.
    """
    # Apply RoPE: shape [L, D]
    xr = apply_rope(x.unsqueeze(0), cos, sin, rope_dim).squeeze(0)

    # Get x at position 0
    x0 = xr[0]  # [D]

    sims = []
    for d in range(xr.size(0)):
        sims.append(F.cosine_similarity(x0, xr[d], dim=0).item())
    return sims

def compute_similarity_curve_indep_query(x_keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rope_dim: int):
    """
    Use an independent Gaussian query at position 0, and a full Gaussian key sequence.
    Return cosine similarities for j=1..L-1 (skip self).
    """
    L, D = x_keys.size()
    # independent query (same distribution as keys)
    q_base = torch.randn(D, device=x_keys.device, dtype=x_keys.dtype)

    # apply RoPE to query at position 0
    # build a fake length-1 tensor with seq_len dimension, then rotate with cos[0], sin[0]
    q0 = apply_rope(q_base.view(1, 1, 1, D),  # shape [B=1,H=1,L=1,D]
                    cos[0].view(1, 1, 1, -1), 
                    sin[0].view(1, 1, 1, -1),
                    rope_dim).view(D)

    # rotate full key sequence with their own positions
    k = apply_rope(x_keys.unsqueeze(0), cos, sin, rope_dim).squeeze(0)  # [L, D]

    sims = []
    for j in range(1, L):  # skip j=0 (self)
        sims.append(F.cosine_similarity(q0, k[j], dim=0).item())
    return sims


def save_curve(curve, filename):
    """
    Save curve as image without displaying.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(curve)
    plt.xlabel("Distance (d)")
    plt.ylabel("Cosine Similarity")
    plt.title(filename)
    plt.grid(True)
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # Settings
    L = 5000
    D = 1024
    device = torch.device("cpu")
    dtype = torch.float32

    # Build RoPE cache
    cos, sin, rope_dim = build_rope_cache(seq_len=L, dim=D, base=10000.0, device=device, dtype=dtype)

    # =============== Experiment 1: Random Vector ===============
    x_random = torch.randn(L, D, device=device, dtype=dtype)
    curve_random = compute_similarity_curve_indep_query(x_random, cos, sin, rope_dim)

    save_curve(curve_random, "rope_decay_random.png")

    # =============== Experiment 2: All-One Vector ===============
    x_one = torch.ones(L, D, device=device, dtype=dtype)
    curve_one = compute_similarity_curve(x_one, cos, sin, rope_dim)
    save_curve(curve_one, "rope_decay_ones.png")

    print("Done! Saved rope_decay_random.png and rope_decay_ones.png")
