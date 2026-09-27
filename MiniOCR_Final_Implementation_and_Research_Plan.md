# MiniOCR — Final End-to-End Implementation & Research Plan

## 1. Project Definition

**Project name:** MiniOCR  
**Goal:** Build a lightweight, custom Vision-Language Model for OCR that is small enough for practical on-device/edge deployment while retaining strong text-recognition capability.

The project should not be presented as simply "a smaller TrOCR." The core research direction is an **OCR-specific, parameter-efficient multimodal architecture** built around:

1. A lightweight ViT vision encoder.
2. Explicit 2D spatially-aware visual tokens.
3. An OCR-specific spatial Q-Former/token compressor.
4. Learned reading-order information.
5. A custom GPT-2-inspired causal multimodal decoder.
6. An OCR-oriented character/BPE tokenizer.
7. A synthetic-to-real OCR training curriculum.
8. Explicit efficiency/accuracy ablation studies.

The starting architecture from the implementation plan is approximately 33.5M parameters, with a 224×224 RGB input, a 6-layer 256-dimensional ViT, 96 Q-Former queries at 384 dimensions, and an 8-layer 384-dimensional decoder. The exact parameter count must be recomputed from the final implementation rather than assumed.

---

# 2. Final Architecture

```text
                         INPUT IMAGE
                       RGB, 224 × 224
                              │
                              ▼
                  ┌──────────────────────┐
                  │      ViT-Tiny        │
                  │  6 Transformer       │
                  │  d_model = 256       │
                  │  4 heads             │
                  │  FFN = 512           │
                  └──────────┬───────────┘
                             │
                       196 × 256
                             │
                             ▼
                  ┌──────────────────────┐
                  │ 2D Spatial Encoding  │
                  │                      │
                  │ row + column         │
                  │ position information │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Spatial OCR         │
                  │ Q-Former             │
                  │                      │
                  │ 96 learned queries  │
                  │ d_model = 384        │
                  │ 3 layers             │
                  └──────────┬───────────┘
                             │
                 ┌───────────┴────────────┐
                 │                        │
                 ▼                        ▼
          OCR visual tokens        Reading-order
             96 × 384                 signals
                 │                        │
                 └───────────┬────────────┘
                             ▼
                 ┌────────────────────────┐
                 │ Custom GPT-OCR Decoder │
                 │                        │
                 │ Causal Self-Attention  │
                 │          ↓             │
                 │ Spatial Cross-Attention│
                 │          ↓             │
                 │ Reading-order bias     │
                 │          ↓             │
                 │ GPT-style MLP          │
                 │                        │
                 │ 8 layers               │
                 │ d_model = 384          │
                 │ 6 heads                │
                 │ FFN = 1536             │
                 └───────────┬────────────┘
                             │
                             ▼
                       LM Head / logits
                             │
                             ▼
                    OCR-specific tokenizer
                             │
                             ▼
                          TEXT OUTPUT
```

## Important terminology

The decoder should be described as **GPT-2-inspired/custom GPT-style**, not "standard GPT-2," because the model must contain image cross-attention and OCR-specific reading-order conditioning. Vanilla GPT-2 is a causal language model and does not natively provide the required visual cross-attention.

Likewise, the connector should be described as an **OCR-specific Spatial Q-Former** or **Q-Former-style bridge** unless its implementation exactly follows the canonical Q-Former design.

---

# 3. Target Model Configuration

## Vision encoder

| Component | Target |
|---|---:|
| Input | 3 × 224 × 224 |
| Patch size | 16 × 16 |
| Patch grid | 14 × 14 |
| Visual tokens | 196 |
| Hidden size | 256 |
| Transformer layers | 6 |
| Attention heads | 4 |
| FFN dimension | 512 |
| Dropout | ~0.1 |

The existing implementation uses a Conv2D patch projection followed by TransformerEncoder layers.

## Spatial representation

Every visual token should contain:

```text
ViT visual feature
+ 2D row information
+ 2D column information
```

The spatial encoding must preserve the original 14×14 geometry rather than treating the image patches as an arbitrary 1D sequence.

Potential future extension:

```text
visual feature
+ x
+ y
+ scale/region information
```

## Spatial OCR Q-Former

| Component | Target |
|---|---:|
| Input | 196 × 256 |
| Projection | 256 → 384 |
| Learned queries | 96 |
| Hidden size | 384 |
| Layers | 3 |
| Heads | 6 |
| FFN | 1024 |

The Q-Former compresses the 196 visual patches into 96 OCR-oriented visual memory tokens.

The goal is not merely compression. It should learn to preserve:

- character-level information,
- local text structure,
- line structure,
- spatial relationships,
- reading order.

## Reading-order module

Each relevant visual token should receive a learned reading-order representation.

A first implementation can predict:

```text
order_score ∈ R
```

from the visual representation.

A stronger implementation can learn a continuous order coordinate and use it inside decoder cross-attention.

## Custom GPT-OCR decoder

| Component | Target |
|---|---:|
| Layers | 8 |
| Hidden size | 384 |
| Heads | 6 |
| FFN | 1536 |
| Attention | causal self-attention + visual cross-attention |
| Activation | GELU |
| Normalization | LayerNorm |
| Position encoding | sinusoidal or learned; compare experimentally |

Each decoder block:

```text
x
 ↓
LayerNorm
 ↓
Causal Self-Attention
 ↓
Residual
 ↓
LayerNorm
 ↓
Spatial / Reading-Order-Aware Cross-Attention
 ↓
Residual
 ↓
LayerNorm
 ↓
MLP
 ↓
Residual
```

Use pre-normalization where practical for training stability.

---

# 4. OCR-Specific Tokenizer

The tokenizer should be designed around OCR rather than generic natural-language modeling.

## Recommended starting design

Use a **character-first BPE / byte-aware tokenizer** with:

- uppercase/lowercase characters,
- digits,
- punctuation,
- whitespace representation,
- common OCR symbols,
- common character sequences,
- special tokens.

Special tokens:

```text
<PAD>
<BOS>
<EOS>
<UNK>
```

Start with a vocabulary around **1,000–2,000 tokens**, then compare against 5,000.

## Why this matters

OCR contains:

- rare names,
- numbers,
- punctuation,
- unusual symbols,
- spelling variants,
- mixed alphanumeric strings.

A pure word-level vocabulary is inappropriate.

A pure character tokenizer is robust but can create long sequences.

A character-first BPE design provides a useful compromise:

```text
frequent sequence → larger token
rare sequence     → smaller token
unknown character → byte/character fallback
```

## Weight tying

Tie the token embedding and LM head when possible:

```python
lm_head.weight = token_embedding.weight
```

This can substantially reduce parameter count.

---

# 5. Reading-Order-Aware Decoder

Reading order is a central OCR-specific contribution.

## Basic concept

For each visual token:

```text
visual feature
+ spatial location
+ learned reading-order coordinate
```

For each generated decoder state:

```text
decoder hidden state
→ expected reading position
```

The cross-attention should use both visual similarity and order compatibility.

Normal:

```text
Attention = softmax(QKᵀ / √d)
```

Proposed:

```text
Attention =
softmax(QKᵀ / √d + B_order)
```

where `B_order` expresses compatibility between the current generated position and the visual token's reading-order position.

A useful formulation is:

```text
d(i,t) = order_i - expected_order_t

B_order(i,t) =
    -(d(i,t)^2) / (2σ²)
```

This gives the decoder a soft preference for visual regions that occur at the appropriate location in the reading sequence.

The order signal should remain **soft**, not a hard sorting operation, because real documents may contain multiple columns, tables, irregular layouts, and ambiguous reading orders.

---

# 6. Reading Order Learning

## Synthetic data advantage

The synthetic data generator knows the locations of every rendered text region.

For example:

```text
HELLO → bounding box A
WORLD → bounding box B
TOTAL → bounding box C
```

Therefore it can automatically produce approximate reading-order labels.

Possible target:

```text
HELLO  → 0.10
WORLD  → 0.50
TOTAL  → 0.90
```

Train an auxiliary order objective:

```text
L_total =
    L_OCR
    + λ_order × L_order
```

Start with a small `λ_order`, then ablate it.

The project should explicitly test whether order supervision improves OCR.

---

# 7. Dataset Strategy

Do not begin with a large real OCR dataset.

Build the training system in stages.

## Stage A — synthetic OCR

Generate text images using Pillow.

Include:

### Easy

- single line,
- clean background,
- standard fonts,
- black text,
- horizontal text.

### Medium

- multiple lines,
- variable font sizes,
- different fonts,
- spacing changes,
- alignment changes,
- grayscale variation.

### Hard

- blur,
- noise,
- compression artifacts,
- rotation,
- perspective distortion,
- shadows,
- uneven illumination,
- textured backgrounds.

### Document-style

- paragraphs,
- receipts,
- invoices,
- forms,
- tables,
- key-value layouts,
- multi-column text,
- headers and footers.

The synthetic generator should also return metadata:

```text
image
text
bounding boxes
line IDs
reading order
```

This metadata enables auxiliary spatial/order training.

---

# 8. Real OCR Data

After synthetic training, fine-tune/evaluate on real OCR datasets.

The exact datasets should be selected based on licensing, language coverage, task scope, and availability.

Recommended evaluation categories:

```text
scene/text crops
documents
receipts
forms
tables
mixed layouts
```

Keep a strict separation between:

```text
training
validation
test
```

Avoid accidentally evaluating on images or near-duplicates used for training.

---

# 9. Image Resolution Strategy

224×224 is convenient for a small model but is a potential OCR bottleneck.

The main risk is:

```text
full document
     ↓
224×224
     ↓
tiny characters
```

Important experiments:

```text
224 × 224
256 × 256
higher-resolution crop
aspect-preserving resize
line/crop OCR
tiling
```

Do not assume that adding more Q-Former queries solves resolution loss.

More queries cannot recover visual information that disappeared during image downsampling.

For the first successful model, target:

> cropped text regions and relatively simple document images.

Then expand toward full-page OCR.

---

# 10. Training Pipeline

## Phase 0 — architecture validation

Before training seriously:

1. Instantiate the complete model.
2. Print parameter counts for every module.
3. Verify tensor shapes.
4. Run a forward pass.
5. Run backward pass.
6. Check for NaNs.
7. Measure GPU memory.
8. Measure inference latency.
9. Confirm mixed-precision training works.

Do not trust the original ~33.5M estimate until the actual implementation reports it.

## Phase 1 — tokenizer

Implement:

```text
tokenizer training
encode()
decode()
special tokens
padding
attention masks
```

Test:

```text
text
→ tokens
→ reconstructed text
```

with Unicode and unusual OCR strings.

## Phase 2 — synthetic data

Build the Pillow generator.

Initially generate:

```text
image + text
```

Then add:

```text
bounding boxes
line IDs
reading order
```

## Phase 3 — tiny overfit test

Train on only:

```text
10 samples
```

and then:

```text
50–100 samples
```

The model should eventually memorize them.

If it cannot, do not scale training.

Check:

- tokenizer,
- causal mask,
- target shifting,
- padding masks,
- cross-attention,
- loss calculation,
- EOS handling,
- image normalization,
- model initialization.

## Phase 4 — synthetic curriculum

Progress:

```text
clean single-line
        ↓
multi-line
        ↓
fonts/sizes
        ↓
noise/distortion
        ↓
multi-region documents
        ↓
tables/forms/receipts
```

## Phase 5 — real-data fine-tuning

Use the synthetic-trained model as initialization.

Evaluate on held-out real OCR data.

---

# 11. Teacher-Forcing Training

Training should use shifted targets.

Example:

```text
Input:
<BOS> HELLO

Target:
HELLO <EOS>
```

For a sequence:

```text
decoder input:
<BOS> H E L L O

target:
H E L L O <EOS>
```

The decoder uses causal masking so position `t` cannot see future target tokens.

Use:

```python
CrossEntropyLoss(ignore_index=PAD_ID)
```

to exclude padding from the loss.

---

# 12. Autoregressive Inference

Inference should work as:

```text
<BOS>
  ↓
GPT decoder
  ↓
token 1
  ↓
token 2
  ↓
token 3
  ↓
...
  ↓
<EOS>
```

Start with greedy decoding:

```text
next_token = argmax(logits)
```

Then optionally add:

- beam search,
- temperature,
- top-k,
- top-p.

For OCR, greedy decoding should be the first baseline because it is simple and efficient.

---

# 13. Efficient Generation

The initial implementation can recompute the decoder sequence at every generation step.

Later, implement a KV cache.

Target:

```text
image encoding → once
Q-Former → once
decoder KV states → cached
```

This is important for practical inference latency.

A custom GPT-style decoder gives more control over implementing this efficiently than a generic high-level TransformerDecoder.

---

# 14. Losses

Primary:

```text
L_OCR = token cross-entropy
```

Optional auxiliary losses:

```text
L_order = reading-order prediction loss
```

Potential future:

```text
L_align = visual/text alignment loss
```

Final experimental formulation:

```text
L_total =
L_OCR
+ λ_order L_order
+ λ_align L_align
```

Do not introduce every auxiliary loss immediately.

Start with:

```text
L_total = L_OCR
```

then add order supervision and measure whether it helps.

---

# 15. Optimization

Starting point:

```text
Optimizer: AdamW
Learning rate: around 2e-4 for initial experiments
Weight decay: tune experimentally
Warmup: recommended
Gradient clipping: recommended
Mixed precision: FP16/BF16 depending on hardware
```

The original implementation uses AdamW at `2e-4`; treat that as an initial hyperparameter, not a guaranteed optimal value.

For modern PyTorch, use the current AMP APIs appropriate to the installed version.

Monitor:

- training loss,
- validation loss,
- CER,
- WER,
- exact match,
- gradient norm,
- GPU memory.

---

# 16. Evaluation Metrics

The main OCR metrics should be:

## Character Error Rate

```text
CER = edit_distance(prediction, target) / target_length
```

## Word Error Rate

```text
WER = word-level edit distance / target word count
```

## Exact Match

Percentage of samples for which:

```text
prediction == target
```

## Efficiency

Measure:

- total parameters,
- trainable parameters,
- model file size,
- FLOPs/MACs where practical,
- peak training VRAM,
- inference VRAM,
- CPU latency,
- GPU latency,
- tokens/sec.

Never claim a memory target such as "under 2.5 GB" without measuring it with the final training configuration.

---

# 17. Core Ablation Study

This is essential to establish that the proposed components actually matter.

## Architecture ablation

### Model A — baseline

```text
ViT → generic Q-Former → GPT-style decoder
```

### Model B — spatial

```text
ViT + 2D spatial tokens
→ Q-Former
→ GPT decoder
```

### Model C — spatial + order

```text
spatial ViT
→ Spatial Q-Former
→ reading-order-aware GPT decoder
```

### Model D — full MiniOCR

```text
spatial ViT
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

This table should become one of the central results of the BTP.

---

# 18. Query-Count Ablation

Test:

```text
32 queries
64 queries
96 queries
128 queries
```

Measure:

- CER,
- WER,
- exact match,
- latency,
- memory.

Research question:

> How much visual-token bandwidth does lightweight OCR require?

---

# 19. Decoder Ablation

Compare:

```text
4 layers
6 layers
8 layers
```

while keeping the other major settings fixed.

Research question:

> What is the accuracy/efficiency tradeoff of decoder depth?

---

# 20. Vision Ablation

Compare:

```text
4 ViT layers
6 ViT layers
8 ViT layers
```

if resources permit.

Also test whether increasing vision capacity produces more benefit than increasing decoder capacity.

---

# 21. Tokenizer Ablation

Compare:

```text
character tokenizer
small BPE tokenizer
character-first BPE/byte-aware tokenizer
```

Measure:

- sequence length,
- CER,
- WER,
- exact match,
- parameter count,
- inference speed.

---

# 22. Reading-Order Ablation

Compare:

```text
No order information
        vs
2D position only
        vs
2D + learned order score
        vs
2D + order-aware cross-attention
```

This directly tests the proposed novel component.

---

# 23. Resolution Ablation

Compare:

```text
224×224
256×256
higher resolution / crops
```

and document whether accuracy improves enough to justify the additional compute.

---

# 24. Robustness Evaluation

Create controlled test sets for:

- blur,
- Gaussian noise,
- JPEG compression,
- rotation,
- perspective distortion,
- low contrast,
- shadows,
- different fonts,
- small text,
- long text,
- mixed alphanumeric strings,
- punctuation-heavy text.

Report CER/WER separately for each category.

---

# 25. Qualitative Error Analysis

For incorrect predictions, classify errors:

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

Examples:

```text
O ↔ 0
I ↔ l ↔ 1
S ↔ 5
B ↔ 8
```

This will help determine whether the architecture is actually solving OCR-specific problems.

---

# 26. Project Repository

Recommended structure:

```text
MiniOCR/
│
├── README.md
├── requirements.txt
├── config/
│   └── base.yaml
│
├── models/
│   ├── vision.py
│   ├── spatial.py
│   ├── qformer.py
│   ├── attention.py
│   ├── gpt_ocr.py
│   └── miniocr.py
│
├── tokenizer/
│   ├── train_tokenizer.py
│   ├── tokenizer.py
│   └── vocab/
│
├── data/
│   ├── synthetic.py
│   ├── dataset.py
│   ├── collator.py
│   └── preprocessing.py
│
├── training/
│   ├── train.py
│   ├── losses.py
│   ├── scheduler.py
│   └── checkpoint.py
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate.py
│   └── error_analysis.py
│
├── inference/
│   ├── generate.py
│   └── benchmark.py
│
├── experiments/
│   ├── query_ablation.py
│   ├── decoder_ablation.py
│   ├── tokenizer_ablation.py
│   ├── order_ablation.py
│   └── resolution_ablation.py
│
├── notebooks/
│
└── checkpoints/
```

---

# 27. Implementation Order

Do not implement everything simultaneously.

## Milestone 1 — Clean architecture

Implement:

```text
PatchEmbedding
SpatialViTEncoder
SpatialPositionEmbedding
SpatialOCRQFormer
ReadingOrderHead
CausalSelfAttention
SpatialCrossAttention
GPTOCRBlock
GPTOCRDecoder
MiniOCR
```

Then verify shapes and parameter counts.

## Milestone 2 — Tokenizer

Implement and validate:

```text
train tokenizer
encode
decode
special tokens
padding
```

## Milestone 3 — Dataset

Implement synthetic Pillow generation with:

```text
image
text
boxes
line IDs
reading order
```

## Milestone 4 — Overfit

Train on 10–100 samples.

Target:

```text
near-zero training CER
```

before scaling.

## Milestone 5 — Synthetic training

Train on increasingly difficult synthetic data.

## Milestone 6 — Real OCR

Fine-tune on real OCR datasets.

## Milestone 7 — Evaluation

Run:

```text
CER
WER
exact match
latency
memory
parameter count
```

## Milestone 8 — Ablations

Run the architecture experiments.

## Milestone 9 — Deployment

Export and benchmark the final model.

Possible targets:

```text
PyTorch
TorchScript / appropriate modern export path
ONNX
mobile/edge runtime if feasible
```

---

# 28. Deployment Goal

The final model should be evaluated as an edge-oriented model.

Report:

```text
Parameter count
Model size
Peak RAM/VRAM
CPU inference time
GPU inference time
Tokens/sec
```

Do not make an "on-device" claim solely because the model has fewer than 100M parameters.

Actual deployment suitability depends on:

- memory,
- compute,
- latency,
- runtime compatibility,
- input resolution,
- sequence length.

---

# 29. Expected Research Contributions

The strongest defensible contributions are:

### Contribution 1
A lightweight OCR-oriented VLM architecture combining a compact ViT, visual token compression, and a custom causal decoder.

### Contribution 2
Explicit **2D spatial awareness** in the visual representation.

### Contribution 3
A **reading-order-aware cross-attention mechanism** for autoregressive OCR generation.

### Contribution 4
An **OCR-specific character/BPE tokenizer** designed for compact vocabulary and robust rare-character handling.

### Contribution 5
A systematic study of the accuracy/efficiency trade-offs of:

- visual-token count,
- decoder depth,
- tokenizer design,
- image resolution,
- spatial information,
- reading-order conditioning.

The fifth contribution is particularly important because it converts architectural ideas into measurable research.

---

# 30. What Should NOT Be Claimed

Do not claim:

- that ViT + Q-Former + causal decoder is entirely novel;
- that GPT-2 itself is novel;
- that Q-Former itself is novel;
- that the model is automatically better than TrOCR;
- that 224×224 is sufficient for all full-page OCR;
- that the model uses <2.5 GB VRAM without measuring it;
- that the architecture is production-ready before deployment testing.

The novelty should instead be framed as:

> **An aggressively parameter-efficient OCR-specific architecture that combines spatial visual tokenization, learned reading-order conditioning, and a lightweight GPT-inspired multimodal decoder, evaluated through systematic accuracy-efficiency ablations.**

---

# 31. Comparison With TrOCR

The project should explicitly acknowledge that TrOCR establishes the broader paradigm of a Transformer-based image encoder paired with an autoregressive text Transformer.

Therefore:

```text
TrOCR-like baseline:
Vision Transformer
      ↓
Text Transformer decoder
      ↓
Text
```

MiniOCR:

```text
OCR-specific lightweight architecture:
Vision Transformer
      ↓
2D spatial representation
      ↓
OCR-specific visual compression
      ↓
reading-order representation
      ↓
GPT-inspired multimodal decoder
      ↓
OCR-specific tokenizer
      ↓
Text
```

The goal is not to claim that the high-level encoder-decoder idea is new.

The goal is to investigate whether OCR-specific structure can provide a better **accuracy / parameter / latency trade-off** in a very small model.

---

# 32. Immediate First Implementation

The first coding target should be a single runnable model:

```python
model = MiniOCR(
    vocab_size=2000,
    image_size=224,
    patch_size=16,
    vision_dim=256,
    vision_depth=6,
    qformer_queries=96,
    qformer_dim=384,
    qformer_depth=3,
    decoder_dim=384,
    decoder_depth=8,
    decoder_heads=6,
    decoder_ffn=1536,
)
```

It must support:

```python
logits = model(
    images,
    input_ids
)
```

and return:

```text
[B, sequence_length, vocab_size]
```

It should also expose:

```python
visual_features
ocr_tokens
order_scores
decoder_hidden_states
```

during debugging/analysis.

---

# 33. Final Success Criteria

The project is successful only if all of the following are demonstrated:

### Engineering

- complete end-to-end training,
- stable loss,
- successful autoregressive inference,
- reproducible checkpoints,
- measured resource usage.

### OCR

- low CER,
- low WER,
- strong exact-match performance,
- robustness across image conditions.

### Research

- baseline comparison,
- spatial ablation,
- reading-order ablation,
- tokenizer ablation,
- query-count ablation,
- decoder-depth ablation,
- resolution ablation.

### Efficiency

- parameter count,
- model size,
- memory,
- latency,
- throughput.

### Novelty

The strongest evidence should show that one or more OCR-specific components provide a measurable benefit **without destroying the lightweight nature of the model**.

---

# 34. Final Development Sequence

```text
                ┌───────────────────────┐
                │ Freeze architecture   │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Implement model       │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Parameter/shape audit │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Build OCR tokenizer   │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Synthetic generator   │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ 10–100 sample overfit │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Synthetic curriculum  │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Real OCR fine-tuning  │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Baseline comparison   │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Ablation experiments  │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Efficiency benchmark  │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Error analysis        │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ Deployment prototype  │
                └──────────┬────────────┘
                           ↓
                ┌───────────────────────┐
                │ BTP report/paper      │
                └───────────────────────┘
```

# 35. Final Recommended Model

**MiniOCR-SOT**

> **S**patial visual tokens  
> **O**rder-aware decoding  
> **T**okenizer optimized for OCR

```text
224×224 Image
      ↓
6-layer ViT-Tiny
256 dim
      ↓
2D spatial encoding
      ↓
3-layer Spatial OCR Q-Former
96 × 384
      ↓
Reading-order representation
      ↓
8-layer GPT-inspired multimodal decoder
384 dim / 6 heads / FFN 1536
      ↓
OCR-specific character-first BPE tokenizer
~1k–2k initial vocabulary
      ↓
Autoregressive OCR text
```

This should be treated as the **final target architecture**, while the baseline and ablation models are kept deliberately simpler so that the project can demonstrate which additions actually provide value.
