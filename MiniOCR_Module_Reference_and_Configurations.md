# MiniOCR — Module Reference & Configurations

## 1. Overall Architecture

```text
Image (RGB 224×224)
        ↓
ViT-Tiny Vision Encoder
        ↓
2D Spatial Encoding
        ↓
Spatial OCR Q-Former
        ↓
Reading-Order Representation
        ↓
Custom GPT-inspired OCR Decoder
        ↓
LM Head
        ↓
OCR-specific Tokenizer
        ↓
Text
```

The target model is approximately 33M parameters. The exact final count must be measured from the implemented model.

---

## 2. Input Preprocessing

**Purpose:** Convert input images into normalized tensors.

| Setting | Target |
|---|---:|
| Channels | 3 |
| Base resolution | 224×224 |
| Format | RGB |
| Tensor | `[B, 3, 224, 224]` |

224×224 keeps the model small but may lose fine text detail on full-page documents, so resolution/cropping should be evaluated.

---

## 3. Patch Embedding

**Purpose:** Divide the image into patches and convert each patch into a feature vector.

For 224×224 with 16×16 patches:

```text
224 / 16 = 14
14 × 14 = 196 patches
```

| Setting | Target |
|---|---:|
| Patch size | 16×16 |
| Grid | 14×14 |
| Patches | 196 |
| Embedding dimension | 256 |

Typical implementation:

```python
nn.Conv2d(3, 256, kernel_size=16, stride=16)
```

Output:

```text
[B, 196, 256]
```

---

## 4. ViT-Tiny Vision Encoder

**Purpose:** Learn visual representations from the 196 image patches.

| Setting | Target |
|---|---:|
| Layers | 6 |
| Hidden dimension | 256 |
| Attention heads | 4 |
| FFN dimension | 512 |
| Input tokens | 196 |
| Dropout | ~0.1 |
| Normalization | LayerNorm |

Structure:

```text
Patch embeddings
      ↓
Positional information
      ↓
Transformer Encoder × 6
      ↓
[B, 196, 256]
```

Each block contains self-attention, residual connections, LayerNorm, and an FFN.

---

## 5. 2D Spatial Encoding

**Purpose:** Explicitly preserve where each visual patch occurs in the image.

For the 14×14 grid:

| Setting | Target |
|---|---:|
| Rows | 14 |
| Columns | 14 |
| Spatial dimension | 256 |
| Row embeddings | 14×256 |
| Column embeddings | 14×256 |

Conceptually:

```text
spatial token =
    visual feature
  + row embedding
  + column embedding
```

Output remains:

```text
[B, 196, 256]
```

Possible future extensions include x/y coordinates and region/scale information.

---

## 6. Visual Projection

**Purpose:** Map ViT features into the multimodal dimension used by the Q-Former and decoder.

```text
256 → 384
```

Typical layer:

```python
nn.Linear(256, 384)
```

Output:

```text
[B, 196, 384]
```

---

## 7. Spatial OCR Q-Former

**Purpose:** Compress 196 spatial visual features into a smaller set of OCR-oriented visual memory tokens.

| Setting | Target |
|---|---:|
| Input tokens | 196 |
| Input dimension | 256 |
| Projection | 256→384 |
| Learned queries | 96 |
| Hidden dimension | 384 |
| Layers | 3 |
| Heads | 6 |
| FFN | 1024 |
| Output | `[B, 96, 384]` |

Concept:

```text
96 learned OCR queries
          ↓
self-attention
          ↓
cross-attention to visual features
          ↓
96 OCR visual tokens
```

This is a **Q-Former-style OCR connector**, not necessarily a canonical BLIP-2 Q-Former.

---

## 8. Learned Query Tokens

**Purpose:** Provide the Q-Former with trainable information-extraction queries.

Initial tensor:

```text
[1, 96, 384]
```

Expanded per batch:

```text
[B, 96, 384]
```

The queries learn to extract useful information from different image regions/features.

---

## 9. Reading-Order Head

**Purpose:** Give the visual representation an explicit soft reading-order signal.

Basic design:

```text
384-d visual token
       ↓
Linear layer
       ↓
1 scalar order score
```

Output:

```text
[B, 96]
```

The score represents a continuous reading-order coordinate rather than a hard sorting index.

---

## 10. Reading-Order-Aware Cross-Attention

**Purpose:** Let the decoder consider both visual similarity and reading-order compatibility.

Normal attention:

```text
softmax(QKᵀ / √d)
```

Proposed:

```text
softmax(QKᵀ / √d + B_order)
```

`B_order` represents compatibility between the decoder's expected reading position and each visual token's learned order position.

A useful starting formulation:

```text
difference = visual_order - expected_order

B_order =
    -(difference²) / (2σ²)
```

The order information should be soft rather than a hard sort because documents can contain multiple columns, tables, and irregular layouts.

---

## 11. OCR-Specific Tokenizer

**Purpose:** Represent OCR text efficiently while handling rare characters and symbols.

### Recommended design

**Character-first BPE / byte-aware tokenizer**

Support:

- letters
- digits
- punctuation
- whitespace
- OCR symbols
- frequent character sequences
- rare/unusual characters

Special tokens:

```text
<PAD>
<BOS>
<EOS>
<UNK>
```

### Initial vocabulary

```text
~1,000–2,000 tokens
```

Compare against a 5,000-token vocabulary.

### Why?

For `d_model=384`:

```text
5000 × 384 = 1.92M
2000 × 384 = 0.768M
```

A smaller vocabulary can therefore reduce parameters, although sequences may become longer.

---

## 12. Token Embedding

**Purpose:** Convert token IDs into vectors for the GPT-style decoder.

| Setting | Target |
|---|---:|
| Vocabulary | 1,000–2,000 initially |
| Embedding dimension | 384 |
| Padding | `<PAD>` |

Output:

```text
[B, T, 384]
```

---

## 13. Text Positional Encoding

**Purpose:** Tell the decoder the order of generated tokens.

Initial choice:

```text
sinusoidal positional encoding
```

Alternative:

```text
learned positional embeddings
```

Use a maximum sequence length appropriate to the OCR task, initially around 128–256 tokens.

---

## 14. Custom GPT-OCR Decoder

**Purpose:** Generate OCR text autoregressively while conditioning on image features.

It is **GPT-2-inspired/custom GPT-style**, not vanilla GPT-2, because it includes visual cross-attention and reading-order conditioning.

| Setting | Target |
|---|---:|
| Layers | 8 |
| Hidden dimension | 384 |
| Heads | 6 |
| Head dimension | 64 |
| FFN | 1536 |
| Activation | GELU |
| Normalization | LayerNorm |
| Self-attention | Causal |
| Cross-attention | Visual |

---

## 15. GPT-OCR Decoder Block

Each block:

```text
Input
  ↓
LayerNorm
  ↓
Causal Self-Attention
  ↓
Residual
  ↓
LayerNorm
  ↓
Spatial/Order-Aware Cross-Attention
  ↓
Residual
  ↓
LayerNorm
  ↓
MLP
  ↓
Residual
```

Repeat for 8 layers.

---

## 16. Causal Self-Attention

**Purpose:** Model relationships among previously generated tokens.

At position `t`, future tokens must be hidden.

| Setting | Target |
|---|---:|
| Hidden dimension | 384 |
| Heads | 6 |
| Head dimension | 64 |
| Mask | Causal |

---

## 17. Visual Cross-Attention

**Purpose:** Allow each generated text state to retrieve relevant image information.

```text
Text hidden state
       ↓
Cross-attention
       ↑
96 OCR visual tokens
```

The decoder therefore combines:

```text
Self-attention → previous text
Cross-attention → image
```

The cross-attention is additionally modified with the reading-order bias.

---

## 18. GPT-Style MLP / FFN

**Purpose:** Nonlinear transformation inside every decoder block.

Configuration:

```text
384
 ↓
1536
 ↓
GELU
 ↓
384
```

Equivalent:

```python
nn.Linear(384, 1536)
nn.GELU()
nn.Linear(1536, 384)
```

---

## 19. LM Head

**Purpose:** Convert decoder hidden states into token logits.

```text
[B, T, 384]
      ↓
Linear(384, vocab_size)
      ↓
[B, T, vocab_size]
```

Recommended:

```python
lm_head.weight = token_embedding.weight
```

Weight tying reduces parameter count.

---

## 20. Autoregressive Generation

**Purpose:** Generate OCR text one token at a time.

```text
<BOS>
  ↓
decoder
  ↓
token 1
  ↓
decoder
  ↓
token 2
  ↓
...
  ↓
<EOS>
```

Start with greedy decoding:

```python
next_token = logits[:, -1].argmax(dim=-1)
```

Later evaluate beam search if useful.

---

## 21. Training Loss

Primary objective:

```text
L_OCR = token cross-entropy
```

Use:

```python
CrossEntropyLoss(ignore_index=PAD_ID)
```

If explicit order supervision is added:

```text
L_total =
    L_OCR
    + λ_order × L_order
```

Start with OCR loss alone, then test the order objective as an ablation.

---

## 22. Teacher Forcing

Training uses shifted sequences.

Example:

```text
Decoder input:
<BOS> H E L L O

Target:
H E L L O <EOS>
```

The causal mask prevents future target tokens from being visible.

---

## 23. Synthetic OCR Dataset

**Purpose:** Provide scalable training data and automatic spatial/order labels.

Each generated example can contain:

```text
image
text
bounding boxes
line IDs
reading order
```

Curriculum:

```text
clean single-line
      ↓
multi-line
      ↓
different fonts/sizes
      ↓
noise/blur/compression
      ↓
rotation/perspective
      ↓
receipts/forms/tables
      ↓
complex documents
```

Pillow can be used for the initial generator.

---

## 24. Data Collator

**Purpose:** Turn variable-length samples into training batches.

Responsibilities:

- image batching
- sequence padding
- attention masks
- shifted decoder inputs
- labels

Outputs:

```text
images
input_ids
attention_mask
labels
```

---

## 25. Training Pipeline

### Phase 0 — Architecture validation

- Instantiate complete model.
- Verify tensor shapes.
- Count parameters.
- Run forward/backward.
- Check for NaNs.
- Measure memory.
- Test mixed precision.

### Phase 1 — Tokenizer

- Train tokenizer.
- Test encode/decode.
- Test special tokens.
- Test Unicode/rare OCR strings.

### Phase 2 — Synthetic data

Build the generator and metadata pipeline.

### Phase 3 — Overfit test

Train on only 10–100 samples.

The model should approach near-zero training CER. If it cannot, debug before scaling.

### Phase 4 — Synthetic curriculum

Progress from clean text to difficult document images.

### Phase 5 — Real OCR

Fine-tune/evaluate on held-out real OCR data.

---

## 26. Real OCR Evaluation

Evaluate across categories such as:

```text
text crops
documents
receipts
forms
tables
mixed layouts
```

Dataset selection should consider licensing, language coverage, task scope, and availability.

Keep training, validation, and test sets strictly separated.

---

## 27. Image Resolution Strategy

224×224 may be insufficient for small text on full pages.

Test:

```text
224×224
256×256
higher-resolution crops
aspect-preserving resize
tiling
line/crop OCR
```

More Q-Former queries cannot recover character information lost through excessive downsampling.

---

## 28. Optimization

Initial configuration:

| Setting | Starting point |
|---|---:|
| Optimizer | AdamW |
| Learning rate | ~2e-4 |
| Weight decay | Tune |
| Warmup | Recommended |
| Precision | FP16/BF16 depending on hardware |
| Gradient clipping | Recommended |

The learning rate is an initial experiment, not a guaranteed optimum.

---

## 29. Evaluation Metrics

### Accuracy

- CER — Character Error Rate
- WER — Word Error Rate
- Exact Match

### Efficiency

- parameter count
- model size
- FLOPs/MACs where practical
- peak training memory
- inference memory
- CPU latency
- GPU latency
- tokens/sec

All efficiency claims should be measured on the final implementation.

---

## 30. Core Ablation Study

### A — Baseline

```text
ViT → Q-Former → GPT decoder
```

### B — Spatial

```text
ViT + 2D spatial encoding
→ Q-Former
→ GPT decoder
```

### C — Spatial + Order

```text
Spatial ViT
→ Spatial Q-Former
→ reading-order representation
→ GPT decoder
```

### D — Full MiniOCR

```text
Spatial ViT
→ Spatial OCR Q-Former
→ reading-order-aware GPT decoder
→ OCR tokenizer
```

Compare:

| Model | Spatial | Order | OCR tokenizer | CER | WER | Params | Latency |
|---|---|---|---|---:|---:|---:|---:|
| A | No | No | No | | | | |
| B | Yes | No | No | | | | |
| C | Yes | Yes | No | | | | |
| D | Yes | Yes | Yes | | | | |

---

## 31. Additional Ablations

### Q-Former query count

```text
32 / 64 / 96 / 128
```

### Decoder depth

```text
4 / 6 / 8 layers
```

### Vision depth

```text
4 / 6 / 8 layers
```

### Tokenizer

```text
character
small BPE
character-first BPE / byte-aware
```

### Reading order

```text
none
2D only
2D + order score
2D + order-aware cross-attention
```

### Resolution

```text
224×224
256×256
higher-resolution/crops
```

Measure accuracy and efficiency for every meaningful comparison.

---

## 32. Robustness Tests

Create controlled test sets for:

- blur
- Gaussian noise
- JPEG compression
- rotation
- perspective distortion
- low contrast
- shadows
- different fonts
- small text
- long text
- mixed alphanumeric strings
- punctuation-heavy text

Report CER/WER separately.

---

## 33. Qualitative Error Analysis

Classify errors into:

```text
character confusion
missing character
extra character
wrong word
wrong reading order
layout confusion
punctuation error
digit error
```

This helps determine whether the proposed architecture addresses OCR-specific failure modes.

---

## 34. Efficient Inference

Initial implementation:

```text
greedy autoregressive decoding
```

Later:

```text
KV cache
```

Cache:

```text
image encoding → once
Q-Former → once
decoder key/value states → reused
```

This is important for practical latency.

---

## 35. Repository Structure

```text
MiniOCR/
├── README.md
├── requirements.txt
├── config/
│   └── base.yaml
├── models/
│   ├── vision.py
│   ├── spatial.py
│   ├── qformer.py
│   ├── attention.py
│   ├── gpt_ocr.py
│   └── miniocr.py
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── tokenizer.py
│   └── vocab/
├── data/
│   ├── synthetic.py
│   ├── dataset.py
│   ├── collator.py
│   └── preprocessing.py
├── training/
│   ├── train.py
│   ├── losses.py
│   └── checkpoint.py
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── error_analysis.py
├── inference/
│   ├── generate.py
│   └── benchmark.py
└── experiments/
    ├── query_ablation.py
    ├── decoder_ablation.py
    ├── tokenizer_ablation.py
    ├── order_ablation.py
    └── resolution_ablation.py
```

---

## 36. Recommended Initial Model Configuration

```text
INPUT
  224×224 RGB

VISION
  patch_size = 16
  patches = 196
  dim = 256
  layers = 6
  heads = 4
  FFN = 512

SPATIAL
  grid = 14×14
  row/column embeddings = 256

Q-FORMER
  queries = 96
  dim = 384
  layers = 3
  heads = 6
  FFN = 1024

ORDER
  learned continuous order representation
  soft order-aware cross-attention

DECODER
  layers = 8
  dim = 384
  heads = 6
  head_dim = 64
  FFN = 1536
  activation = GELU
  causal self-attention = yes
  visual cross-attention = yes

TOKENIZER
  character-first BPE / byte-aware
  vocab = 1,000–2,000 initially
  special tokens = PAD/BOS/EOS/UNK

POSITION
  sinusoidal initially

OUTPUT
  tied embedding/LM head
  autoregressive generation
```

---

## 37. Module-to-Role Summary

| Module | Main role |
|---|---|
| Input preprocessing | Prepare images |
| Patch Embedding | Convert image to patches |
| ViT | Learn visual features |
| 2D Spatial Encoding | Preserve geometry |
| Visual Projection | Match dimensions |
| Q-Former | Compress/select OCR-relevant visual information |
| Learned Queries | Extract visual information |
| Reading-Order Head | Estimate visual reading order |
| Order-Aware Attention | Guide attention toward appropriate regions |
| Tokenizer | Represent OCR text |
| Token Embedding | Convert tokens to vectors |
| Causal Self-Attention | Model previous text |
| Visual Cross-Attention | Retrieve image information |
| GPT MLP | Nonlinear transformation |
| LM Head | Predict next token |
| Loss | Train recognition/order |
| Generator | Produce text |
| Evaluation | Measure accuracy/efficiency |

---

## 38. Final Research Framing

The project should not claim that the high-level image-encoder + autoregressive-text-decoder paradigm is new.

Instead, the research contribution is:

> An aggressively parameter-efficient OCR-specific architecture combining spatial visual tokenization, learned reading-order conditioning, and a lightweight GPT-inspired multimodal decoder, evaluated through systematic accuracy-efficiency ablations.

The central questions are:

1. Does explicit 2D spatial information improve compact OCR?
2. Does reading-order conditioning improve multi-line/structured OCR?
3. Does an OCR-specific tokenizer improve the parameter/sequence-length trade-off?
4. How many visual tokens are actually required?
5. What is the best accuracy/latency/parameter trade-off for a small OCR VLM?

---

## 39. Final Target: MiniOCR-SOT

**S** = Spatial  
**O** = Order-aware  
**T** = OCR-specific Tokenizer

```text
224×224 Image
      ↓
ViT-Tiny
6 layers × 256
      ↓
2D Spatial Encoding
      ↓
Spatial OCR Q-Former
3 layers × 384
96 queries
      ↓
Reading-Order Representation
      ↓
8-layer GPT-inspired OCR Decoder
384 dim / 6 heads / FFN 1536
      ↓
Tied LM Head
      ↓
OCR Character-first BPE / Byte-aware Tokens
      ↓
Autoregressive Text
```
