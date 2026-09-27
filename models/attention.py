"""Causal self-attention and reading-order-aware spatial cross-attention."""

import torch
import torch.nn as nn
import math


class CausalSelfAttention(nn.Module):
    """Causal self-attention with pre-normalization for the GPT-OCR decoder."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        if causal_mask is None:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)

        attn_out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x),
                                attn_mask=causal_mask, need_weights=False)
        return x + attn_out


class SpatialCrossAttention(nn.Module):
    """Cross-attention between decoder states and OCR visual tokens with reading-order bias.

    The decoder is given a soft preference for visual regions whose predicted
    reading-order score matches its current generation step.
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

        self.log_sigma = nn.Parameter(torch.tensor(math.log(sigma)))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        decoder_state: torch.Tensor,
        visual_tokens: torch.Tensor,
        order_scores: torch.Tensor,
        expected_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = decoder_state.shape
        N = visual_tokens.shape[1]
        sigma = self.log_sigma.exp()

        if expected_order is None:
            expected_order = torch.linspace(0.0, 1.0, T, device=decoder_state.device)

        q = self.norm(self.W_q(decoder_state))
        k = self.norm(self.W_k(visual_tokens))
        v = self.norm(self.W_v(visual_tokens))

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # Order bias: prefer attending to visual tokens whose predicted order
        # matches the current decoder position.
        diff = order_scores.unsqueeze(-1) - expected_order.unsqueeze(0).unsqueeze(1)  # [B, 96, T]
        order_bias = -(diff ** 2) / (2.0 * sigma ** 2)
        order_bias = order_bias.transpose(1, 2).unsqueeze(1)  # [B, 1, T, 96]
        order_bias = order_bias.expand(-1, self.n_heads, -1, -1)
        attn = attn + order_bias

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return decoder_state + self.out_proj(out)
