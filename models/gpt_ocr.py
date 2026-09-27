"""GPT-OCR decoder: 8-layer causal multimodal decoder with visual cross-attention."""

import math

import torch
import torch.nn as nn

from .attention import CausalSelfAttention, SpatialCrossAttention


class GPTOCRBlock(nn.Module):
    """A single GPT-OCR decoder block with pre-normalization.

    Each block applies:
        1. Causal self-attention (previous tokens only)
        2. Spatial/reading-order-aware cross-attention to visual tokens
        3. GELU MLP

    Shape:
        input : [B, T, D]
        output: [B, T, D]
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, d_ff: int = 1536,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Causal self-attention
        self.causal_attn = CausalSelfAttention(d_model, n_heads, dropout)

        # Spatial/order-aware cross-attention to visual tokens
        self.cross_attn = SpatialCrossAttention(d_model, n_heads, dropout)

        # GPT-style MLP / FFN
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        visual_tokens: torch.Tensor,
        order_scores: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
        expected_order: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, D] decoder states
            visual_tokens: [B, 96, D] OCR visual memory
            order_scores: [B, 96] reading-order coordinates
            causal_mask: [T, T] causal mask
            expected_order: [T] expected order positions
        Returns:
            [B, T, D] updated decoder states
        """
        # 1. Causal self-attention
        x = self.causal_attn(x, causal_mask)

        # 2. Spatial/order-aware cross-attention
        x = self.cross_attn(x, visual_tokens, order_scores, expected_order)

        # 3. MLP
        x = x + self.mlp(self.norm3(x))
        return x


class GPTOCRDecoder(nn.Module):
    """8-layer GPT-inspired multimodal OCR decoder.

    - Causal self-attention for autoregressive text generation
    - Visual cross-attention conditioned on OCR tokens and reading order
    - GELU activations, pre-normalization, residual connections

    Shape:
        input : [B, T, D]  (token embeddings + positional encoding)
        output: [B, T, D]  (decoder hidden states)
    """

    def __init__(self, d_model: int = 384, n_heads: int = 6, d_ff: int = 1536,
                 num_layers: int = 8, dropout: float = 0.1, max_seq_len: int = 256):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Sinusoidal positional encoding
        self.pos_embed = self._build_sinusoidal(max_seq_len, d_model)

        # Stack of decoder blocks
        self.layers = nn.ModuleList([
            GPTOCRBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def _build_sinusoidal(self, max_len: int, d_model: int) -> torch.Tensor:
        """Build sinusoidal positional encodings (fixed, not learned)."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        return pe

    def forward(
        self,
        token_embeddings: torch.Tensor,
        visual_tokens: torch.Tensor,
        order_scores: torch.Tensor,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            token_embeddings: [B, T, D] token + positional embeddings
            visual_tokens: [B, 96, D] OCR visual memory from Q-Former
            order_scores: [B, 96] reading-order coordinates
            causal_mask: [T, T] causal mask (optional, built if None)
        Returns:
            [B, T, D] decoder hidden states
        """
        B, T, D = token_embeddings.shape

        # Add positional encoding
        pos = self.pos_embed[:, :T, :].to(token_embeddings.device)
        x = token_embeddings + pos

        # Build expected order positions for cross-attention
        # Evenly spaced over [0, 1] for T generation steps
        expected_order = torch.linspace(0.0, 1.0, T, device=token_embeddings.device)

        for layer in self.layers:
            x = layer(x, visual_tokens, order_scores, causal_mask, expected_order)

        return self.norm(x)

    def build_causal_mask(self, T: int) -> torch.Tensor:
        """Create a lower-triangular causal mask for sequence length T."""
        return torch.triu(torch.ones(T, T, device=next(self.parameters()).device), diagonal=1)
