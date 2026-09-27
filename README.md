# MiniOCR — VLM-based OCR Architecture (~33M Parameters)

A lightweight, custom Vision-Language Model for OCR designed for edge deployment.

## Architecture

```
Image (224×224 RGB)
  → PatchEmbedding (196 × 256)
  → 2D Spatial Encoding
  → ViT-Tiny Encoder (6 layers)
  → Spatial OCR Q-Former (96 × 384)
  → Reading-Order Head
  → GPT-OCR Decoder (8 layers)
  → Tied LM Head
  → OCR Tokenizer
  → Text
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Test architecture and run everything
python run_miniocr.py test

# Or run individual steps:
python run_miniocr.py generate   # Generate synthetic OCR dataset
python run_miniocr.py train      # Train the model
python run_miniocr.py eval       # Evaluate and run ablations
```

## Project Structure

```
miniocr-v1/
├── models/           # Architecture blocks
│   ├── patch_embedding.py
│   ├── vision_encoder.py
│   ├── qformer.py
│   ├── attention.py
│   ├── gpt_ocr.py
│   └── miniocr.py
├── tokenizer/        # OCR-specific tokenizer
├── data/             # Dataset utilities
│   └── synthetic_data/
│       ├── generator.py   # Pillow-based data generator
│       └── dataset.py     # PyTorch Dataset
├── training/         # Training and evaluation
│   ├── train.py
│   └── evaluate.py
├── config/           # Configuration files
└── run_miniocr.py    # Main entry point
```

## Key Features

- **2D Spatial Encoding**: Explicit row+column position information
- **Spatial OCR Q-Former**: Compresses 196 visual tokens into 96 OCR memory tokens
- **Reading-Order-Aware Cross-Attention**: Soft order bias in decoder attention
- **OCR-Specific Tokenizer**: Character-first BPE with 2000 vocabulary
- **~33M Parameters**: Lightweight enough for edge deployment
