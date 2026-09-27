"""Spatial OCR Q-Former and Reading-Order Head."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialOCRQFormer(nn.Module):
    """OCR-specific Q-Former that compresses 196 visual tokens into 96 OCR tokens.

    Architecture:
        - Projects 256-dim visual features to 384-dim
        - Uses 96 learned queries that attend to the visual features (cross-attention)
        - Applies 3 transformer layers of Q-Former blocks
        - Output: 96 × 384 OCR visual memory tokens

    Shape:
        input visual : [B, 196, 256]
        output       : [B, 96, 384]
    """

    def __init__(self, visual_dim: int = 256, qformer_dim: int = 384,
                 num_queries: int = 96, num_layers: int = 3, n_heads: int = 6,
                 d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.num_queries = num_queries
        self.qformer_dim = qformer_dim

        # Visual projection: 256 → 384
        self.visual_proj = nn.Linear(visual_dim, qformer_dim)

        # Learned query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, qformer_dim) * 0.02)

        # Q-Former transformer blocks (cross-attention of queries to visual features)
        self.layers = nn.ModuleList([
            QFormerBlock(qformer_dim, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(qformer_dim)

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            visual_features: [B, 196, 256] from ViT encoder (after spatial encoding)
        Returns:
            ocr_tokens: [B, 96, 384] compressed OCR visual memory
        """
        B = visual_features.shape[0]

        # Project visual features to Q-Former dimension
        visual_proj = self.visual_proj(visual_features)  # [B, 196, 384]

        # Expand learned queries to batch size
        queries = self.queries.expand(B, -1, -1)  # [B, 96, 384]

        # Q-Former blocks: queries attend to visual features
        for layer in self.layers:
            queries = layer(queries, visual_proj)

        return self.norm(queries)


class QFormerBlock(nn.Module):
    """A single Q-Former block: self-attention on queries + cross-attention to visuals."""

    def __init__(self, d_model: int = 384, n_heads: int = 6, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # Cross-attention: queries attend to visual features
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

        # FFN
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [B, 96, 384] Q-Former query tokens
            visual_features: [B, 196, 384] projected visual features
        Returns:
            [B, 96, 384] updated queries
        """
        # Self-attention on queries
        attn_out, _ = self.self_attn(
            self.norm1(queries), self.norm1(queries), self.norm1(queries),
            need_weights=False,
        )
        queries = queries + attn_out  # residual

        # Cross-attention: queries as Q, visuals as K/V
        attn_out, _ = self.cross_attn(
            self.norm2(queries), self.norm2(visual_features), self.norm2(visual_features),
            need_weights=False,
        )
        queries = queries + attn_out  # residual

        # FFN
        queries = queries + self.mlp(self.norm3(queries))  # residual
        return queries


class ReadingOrderHead(nn.Module):
    """Predicts a continuous reading-order score for each OCR visual token.

    Each of the 96 OCR tokens gets a scalar order_score ∈ [0, 1] that represents
    where in the reading sequence that token's text region falls.

    Shape:
        input : [B, 96, 384]
        output: [B, 96]   (order scores)
    """

    def __init__(self, dim: int = 384):
        super().__init__()
        self.order_head = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim // 2, 1),
            nn.Sigmoid(),  # clamp to [0, 1]
        )

    def forward(self, ocr_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ocr_tokens: [B, 96, 384] OCR visual memory tokens
        Returns:
            order_scores: [B, 96] continuous reading-order coordinates
        """
        return self.order_head(ocr_tokens).squeeze(-1)  # [B, 96]
