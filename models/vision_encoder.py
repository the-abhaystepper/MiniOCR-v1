"""ViT-Tiny vision encoder: 6 transformer layers, 256-dim, 4 heads, FFN 512."""

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """A single ViT encoder block with pre-normalization, self-attention, and GELU MLP."""

    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] visual tokens
            src_key_padding_mask: [B, N] boolean mask (True = ignored)
        Returns:
            [B, N, D] updated tokens
        """
        # Self-attention with pre-norm
        attn_out, _ = self.attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out  # residual

        # FFN with pre-norm
        x = x + self.mlp(self.norm2(x))  # residual
        return x


class ViTTinyEncoder(nn.Module):
    """Lightweight ViT encoder: 6 layers, hidden 256, 4 heads, FFN 512.

    Takes 196 visual tokens (from PatchEmbedding) and outputs enhanced
    visual features through 6 transformer blocks.

    Shape:
        input : [B, 196, 256]
        output: [B, 196, 256]
    """

    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 512,
                 num_layers: int = 6, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, 196, 256] visual tokens
        Returns:
            [B, 196, 256] encoded visual features
        """
        for layer in self.layers:
            x = layer(x, src_key_padding_mask)
        return self.norm(x)
