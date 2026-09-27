"""Causal self-attention and reading-order-aware spatial cross-attention."""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """Causal self-attention for the GPT-OCR decoder.

    Prevents position t from attending to future tokens using a causal mask.
    Uses pre-normalization for training stability.

    Shape:
        input : [B, T, D]
        output: [B, T, D]
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] decoder hidden states
            causal_mask: [T, T] boolean mask (True = allowed). If None, built.
        Returns:
            [B, T, D] attended features
        """
        B, T, D = x.shape
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        attn_out, _ = self.attn(
            self.norm(x), self.norm(x), self.norm(x),
            attn_mask=causal_mask,
            need_weights=False,
        )
        return x + attn_out  # residual


class SpatialCrossAttention(nn.Module):
    """Reading-order-aware cross-attention between decoder states and OCR visual tokens.

    Computes attention as:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d) + B_order) * V

    where B_order(i,t) = -(order_i - expected_order_t)^2 / (2σ²)

    This gives the decoder a soft preference for visual regions that appear
    at the appropriate position in the reading sequence.

    Shape:
        decoder_state : [B, T, D]
        visual_tokens : [B, 96, D]
        order_scores  : [B, 96]
    Returns:
        [B, T, D] context vectors
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, dropout: float = 0.1,
                 sigma: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.sigma = sigma

        self.norm = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable sigma for the order bias
        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        order_scores: torch.Tensor,
        expected_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_state: [B, T, D] current decoder hidden states
            visual_tokens: [B, 96, D] OCR visual memory tokens
            order_scores: [B, 96] continuous reading-order coordinates
            expected_order: [T] expected order position for each decoder step.
                            If None, evenly spaced in [0, 1].
        Returns:
            [B, T, D] context vectors with order-aware bias
        """
        B, T, D = decoder_state.shape
        N = visual_tokens.shape[1]  # 96

        sigma = self.log_sigma.exp()

        # Compute expected order positions if not provided
        if expected_order is None:
            expected_order = torch.linspace(0.0, 1.0, T, device=decoder_state.device)  # [T]

        # Normalize
        q = self.norm(self.W_q(decoder_state))   # [B, T, D]
        k = self.norm(self.W_k(visual_tokens))    # [B, 96, D]
        v = self.norm(self.W_v(visual_tokens))    # [B, 96, D]

        # Split heads: [B, T, H, d_head] -> [B, H, T, d_head]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, T, d]
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)   # [B, H, 96, d]
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)   # [B, H, 96, d]

        # Scaled dot-product attention scores
        # attn: [B, H, T, 96]
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Add order bias: B_order(i,t) = -(order_i - expected_order_t)^2 / (2σ²)
        # order_scores: [B, 96] → [B, 96, 1]
        # expected_order: [T] → [B, 1, T]
        # diff: [B, 96, T]
        diff = order_scores.unsqueeze(-1) - expected_order.unsqueeze(0).unsqueeze(1)  # [B, 96, T]
        order_bias = -(diff ** 2) / (2.0 * sigma ** 2)  # [B, 96, T]
        # Reshape for head-split attention: [B, 1, T, 96]
        # q: [B, T, H, d] after transpose → [B, H, T, d]
        # k: [B, 96, H, d] after transpose → [B, H, 96, d]
        # attn: [B, H, T, 96]
        # So order_bias needs to be [B, 1, T, 96] → expanded to [B, H, T, 96]
        order_bias = order_bias.transpose(1, 2).unsqueeze(1)  # [B, 1, T, 96]
        order_bias = order_bias.expand(-1, self.n_heads, -1, -1)  # [B, H, T, 96]
        attn = attn + order_bias  # Additive bias

        # Softmax over key dimension (96)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        out = torch.matmul(attn, v)  # [B, H, T, d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]

        return decoder_state + self.out_proj(out)  # residual
