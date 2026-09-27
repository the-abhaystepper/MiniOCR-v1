"""Full MiniOCR model: ViT → Spatial Q-Former → GPT-OCR Decoder."""

import torch
import torch.nn as nn

from .patch_embedding import PatchEmbedding, SpatialPositionEmbedding
from .vision_encoder import ViTTinyEncoder
from .qformer import SpatialOCRQFormer, ReadingOrderHead
from .gpt_ocr import GPTOCRDecoder


class MiniOCR(nn.Module):
    """Complete MiniOCR architecture (~29M parameters)."""

    def __init__(
        self,
        vocab_size: int = 2000,
        image_size: int = 224,
        patch_size: int = 16,
        vision_dim: int = 256,
        vision_depth: int = 6,
        vision_heads: int = 4,
        vision_ffn: int = 512,
        vision_dropout: float = 0.1,
        grid_size: int = 14,
        qformer_queries: int = 96,
        qformer_dim: int = 384,
        qformer_depth: int = 3,
        qformer_heads: int = 6,
        qformer_ffn: int = 1024,
        qformer_dropout: float = 0.1,
        decoder_dim: int = 384,
        decoder_depth: int = 8,
        decoder_heads: int = 6,
        decoder_ffn: int = 1536,
        decoder_dropout: float = 0.1,
        max_seq_len: int = 256,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        unk_token_id: int = 3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_patches = (image_size // patch_size) ** 2
        self.qformer_queries = qformer_queries
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.unk_token_id = unk_token_id

        # Vision encoder
        self.patch_embed = PatchEmbedding(image_size, patch_size, 3, vision_dim)
        self.spatial_pos_embed = SpatialPositionEmbedding(grid_size, vision_dim)
        self.vision_encoder = ViTTinyEncoder(
            d_model=vision_dim, n_heads=vision_heads, d_ff=vision_ffn,
            num_layers=vision_depth, dropout=vision_dropout,
        )

        # Q-Former and reading-order head
        self.qformer = SpatialOCRQFormer(
            visual_dim=vision_dim, qformer_dim=qformer_dim,
            num_queries=qformer_queries, num_layers=qformer_depth,
            n_heads=qformer_heads, d_ff=qformer_ffn, dropout=qformer_dropout,
        )
        self.reading_order_head = ReadingOrderHead(qformer_dim)

        # Decoder
        self.token_embedding = nn.Embedding(vocab_size, decoder_dim, padding_idx=pad_token_id)
        self.decoder = GPTOCRDecoder(
            d_model=decoder_dim, n_heads=decoder_heads, d_ff=decoder_ffn,
            num_layers=decoder_depth, dropout=decoder_dropout, max_seq_len=max_seq_len,
        )
        self.lm_head = nn.Linear(decoder_dim, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')

    @property
    def device_params(self):
        return next(self.parameters()).device

    def encode_image(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode an image into OCR tokens and reading-order scores."""
        B = images.shape[0]

        visual_features = self.patch_embed(images)
        visual_features = self.spatial_pos_embed(visual_features)
        visual_features = self.vision_encoder(visual_features)

        ocr_tokens = self.qformer(visual_features)
        order_scores = self.reading_order_head(ocr_tokens)

        return ocr_tokens, order_scores, visual_features

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass for training (teacher-forcing).

        Args:
            images: [B, 3, 224, 224]
            input_ids: [B, T] token IDs (BOS + tokens)
        Returns:
            logits: [B, T, vocab_size]
        """
        ocr_tokens, order_scores, _ = self.encode_image(images)
        token_emb = self.token_embedding(input_ids)
        T = input_ids.shape[1]
        causal_mask = self.decoder.build_causal_mask(T)

        hidden_states = self.decoder(token_emb, ocr_tokens, order_scores, causal_mask)
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> torch.Tensor:
        """Autoregressive generation."""
        B = images.shape[0]
        ocr_tokens, order_scores, _ = self.encode_image(images)
        max_len = 1 + max_new_tokens
        causal_mask = self.decoder.build_causal_mask(max_len)
        generated = torch.full((B, 1), self.bos_token_id, device=self.device_params, dtype=torch.long)

        for t in range(max_new_tokens):
            token_emb = self.token_embedding(generated)
            hidden_states = self.decoder(
                token_emb, ocr_tokens, order_scores, causal_mask[:t+1, :t+1],
            )
            logits = self.lm_head(hidden_states[:, -1, :])
            logits = logits / temperature

            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[mask] = float('-inf')
                logits.scatter_(1, sorted_indices, sorted_logits)

            if temperature < 0.01:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == self.eos_token_id).all():
                break

        return generated

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_parameters_by_module(self) -> dict[str, int]:
        counts = {}
        for name, module in self.named_modules():
            if not list(module.children()):
                counts[name] = sum(p.numel() for p in module.parameters())
        return counts
