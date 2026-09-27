"""GPT-OCR decoder: 8-layer causal multimodal decoder with visual cross-attention."""

import math
import torch
import torch.nn as nn

from .attention import CausalSelfAttention, SpatialCrossAttention


class GPTOCRBlock(nn.Module):
    """Single decoder block: causal self-attention → cross-attention → MLP."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, d_ff: int = 1536, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.causal_attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.cross_attn = SpatialCrossAttention(d_model, n_heads, dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, visual_tokens: torch.Tensor,
                order_scores: torch.Tensor, causal_mask: torch.Tensor | None = None,
                expected_order: torch.Tensor | None = None) -> torch.Tensor:
        x = self.causal_attn(x, causal_mask)
        x = self.cross_attn(x, visual_tokens, order_scores, expected_order)
        x = x + self.mlp(self.norm3(x))
        return x


class GPTOCRDecoder(nn.Module):
    """8-layer GPT-inspired multimodal decoder with reading-order-biased cross-attention."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, d_ff: int = 1536,
                 num_layers: int = 8, dropout: float = 0.1, max_seq_len: int = 256):
        super().__init__()
        self.pos_embed = self._build_sinusoidal(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            GPTOCRBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _build_sinusoidal(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, token_embeddings: torch.Tensor, visual_tokens: torch.Tensor,
                order_scores: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = token_embeddings.shape
        x = token_embeddings + self.pos_embed[:, :T, :].to(token_embeddings.device)
        expected_order = torch.linspace(0.0, 1.0, T, device=token_embeddings.device)
        for layer in self.layers:
            x = layer(x, visual_tokens, order_scores, causal_mask, expected_order)
        return self.norm(x)

    def build_causal_mask(self, T: int) -> torch.Tensor:
        return torch.triu(torch.ones(T, T, device=next(self.parameters()).device), diagonal=1)
