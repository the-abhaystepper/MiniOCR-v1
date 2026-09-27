"""Spatial OCR Q-Former and Reading-Order Head."""

import torch
import torch.nn as nn


class SpatialOCRQFormer(nn.Module):
    """Compresses 196 visual tokens into 96 learned OCR query tokens."""

    def __init__(self, visual_dim: int = 256, qformer_dim: int = 384,
                 num_queries: int = 96, num_layers: int = 3, n_heads: int = 6,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, qformer_dim)
        self.queries = nn.Parameter(torch.randn(1, num_queries, qformer_dim) * 0.02)
        self.layers = nn.ModuleList([
            QFormerBlock(qformer_dim, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(qformer_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        visual_proj = self.visual_proj(visual_features)
        queries = self.queries.expand(visual_features.shape[0], -1, -1)
        for layer in self.layers:
            queries = layer(queries, visual_proj)
        return self.norm(queries)


class QFormerBlock(nn.Module):
    """Self-attention on queries + cross-attention to visual features + MLP."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(self.norm1(queries), self.norm1(queries), self.norm1(queries),
                                     need_weights=False)
        queries = queries + attn_out
        attn_out, _ = self.cross_attn(self.norm2(queries), self.norm2(visual_features),
                                      self.norm2(visual_features), need_weights=False)
        queries = queries + attn_out
        queries = queries + self.mlp(self.norm3(queries))
        return queries


class ReadingOrderHead(nn.Module):
    """Predicts a continuous reading-order score per OCR visual token."""

    def __init__(self, dim: int = 384):
        super().__init__()
        self.order_head = nn.Sequential(
            nn.Linear(dim, dim // 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(dim // 2, 1), nn.Sigmoid(),
        )

    def forward(self, ocr_tokens: torch.Tensor) -> torch.Tensor:
        return self.order_head(ocr_tokens).squeeze(-1)
