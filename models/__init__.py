"""MiniOCR model package — architectural blocks for the 33M-parameter VLM-OCR system."""

from .patch_embedding import PatchEmbedding, SpatialPositionEmbedding
from .vision_encoder import ViTTinyEncoder
from .qformer import SpatialOCRQFormer, ReadingOrderHead
from .attention import CausalSelfAttention, SpatialCrossAttention
from .gpt_ocr import GPTOCRBlock, GPTOCRDecoder
from .miniocr import MiniOCR

__all__ = [
    "PatchEmbedding",
    "SpatialPositionEmbedding",
    "ViTTinyEncoder",
    "SpatialOCRQFormer",
    "ReadingOrderHead",
    "CausalSelfAttention",
    "SpatialCrossAttention",
    "GPTOCRBlock",
    "GPTOCRDecoder",
    "MiniOCR",
]
