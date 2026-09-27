"""Evaluation script for MiniOCR — metrics, ablation studies, and analysis."""

import os
import sys
import json
import time
import tempfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.miniocr import MiniOCR
from models.metrics import batch_metrics, character_error_rate, word_error_rate, edit_distance
from tokenizer.tokenizer import OCRTokenizer
from data.dataset import create_train_val_test_loaders


def load_model(checkpoint_path: str, device: torch.device = None) -> tuple[MiniOCR, dict]:
    """Load a trained MiniOCR model from checkpoint.

    Args:
        checkpoint_path: Path to .pt checkpoint file
    Returns:
        (model, config_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = MiniOCR(
        vocab_size=2000,
        image_size=224,
        patch_size=16,
        vision_dim=256,
        vision_depth=6,
        vision_heads=4,
        vision_ffn=512,
        qformer_queries=96,
        qformer_dim=384,
        qformer_depth=3,
        qformer_heads=6,
        qformer_ffn=1024,
        decoder_dim=384,
        decoder_depth=8,
        decoder_heads=6,
        decoder_ffn=1536,
        max_seq_len=256,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate_model(
    model: MiniOCR,
    dataloader: DataLoader,
    tokenizer: OCRTokenizer,
    device: torch.device,
    max_samples: int = 200,
) -> dict:
    """Full evaluation on a dataset.

    Args:
        model: Trained MiniOCR model
        dataloader: DataLoader for evaluation
        tokenizer: OCR tokenizer
        device: torch.device
        max_samples: Max samples to evaluate
    Returns:
        dict with all evaluation metrics
    """
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    pad_id = tokenizer.pad_token_id
    use_amp = device.type == "cuda"
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx * dataloader.batch_size >= max_samples:
            break

        images = batch["image"].to(device)
        texts = batch["text"]

        # Tokenize
        input_ids_list = []
        for text in texts:
            encoded = tokenizer.encode(text, add_bos=True, add_eos=False)
            input_ids_list.append(encoded)

        max_len = max(len(ids) for ids in input_ids_list)
        padded_input = []
        padded_labels = []
        for ids in input_ids_list:
            padded_input.append(ids + [pad_id] * (max_len - len(ids)))
            padded_labels.append(ids[1:] + [tokenizer.eos_token_id] + [pad_id] * (max_len - len(ids)))

        input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

        # Forward
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, input_ids)
            loss = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="mean")(
                logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1)
            )
            total_loss += loss.item() * images.shape[0]

        # Generate
        for i in range(images.shape[0]):
            generated_ids = model.generate(
                images[i:i+1], max_new_tokens=64, temperature=0.01
            )[0]
            pred_text = tokenizer.decode(generated_ids.tolist())
            predictions.append(pred_text)
            targets.append(texts[i])

        batch_count += 1

    avg_loss = total_loss / max(len(predictions), 1)
    metrics = batch_metrics(predictions, targets)
    metrics["loss"] = avg_loss
    metrics["num_predictions"] = len(predictions)

    return metrics


@torch.no_grad()
def run_ablation_study(
    data_dir: str = "data/synthetic_data",
    batch_size: int = 16,
    max_samples: int = 200,
) -> dict:
    """Run architecture ablation studies.

    Compares different model configurations:
        - Full MiniOCR (96 queries, 8 decoder layers)
        - Query count variations (32, 64, 128)
        - Decoder depth variations (4, 6, 8)
        - Latency and efficiency profiling

    Returns:
        dict with ablation results
    """
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloaders = create_train_val_test_loaders(data_dir, batch_size=batch_size)
    val_loader = dataloaders["val"]

    results = {}
    tokenizer = OCRTokenizer(vocab_size=2000)

    print("\nRunning ablation study...")

    # Full MiniOCR (Model D)
    print("  Testing Model D (Full MiniOCR)...")
    model_d = MiniOCR(
        vocab_size=2000, image_size=224, patch_size=16,
        vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
        qformer_queries=96, qformer_dim=384, qformer_depth=3, qformer_heads=6,
        qformer_ffn=1024, decoder_dim=384, decoder_depth=8, decoder_heads=6,
        decoder_ffn=1536, max_seq_len=256,
    ).to(device)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model_d.load_state_dict(ckpt["model_state_dict"])

    metrics_d = evaluate_model(model_d, val_loader, tokenizer, device, max_samples)
    results["D_full"] = {
        "params": model_d.count_parameters(),
        **metrics_d,
    }
    print(f"  Model D CER: {metrics_d['CER']:.4f}, WER: {metrics_d['WER']:.4f}")

    # Test with different query counts
    for n_queries in [32, 64, 128]:
        print(f"  Testing {n_queries} queries...")
        model_q = MiniOCR(
            vocab_size=2000, image_size=224, patch_size=16,
            vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
            qformer_queries=n_queries, qformer_dim=384, qformer_depth=3, qformer_heads=6,
            qformer_ffn=1024, decoder_dim=384, decoder_depth=8, decoder_heads=6,
            decoder_ffn=1536, max_seq_len=256,
        ).to(device)

        metrics_q = evaluate_model(model_q, val_loader, tokenizer, device, max_samples)
        results[f"queries_{n_queries}"] = {
            "params": model_q.count_parameters(),
            **metrics_q,
        }
        print(f"  {n_queries} queries CER: {metrics_q['CER']:.4f}")

    # Test with different decoder depths
    for depth in [4, 6, 8]:
        print(f"  Testing decoder depth={depth}...")
        model_dec = MiniOCR(
            vocab_size=2000, image_size=224, patch_size=16,
            vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
            qformer_queries=96, qformer_dim=384, qformer_depth=3, qformer_heads=6,
            qformer_ffn=1024, decoder_dim=384, decoder_depth=depth, decoder_heads=6,
            decoder_ffn=1536, max_seq_len=256,
        ).to(device)

        metrics_dec = evaluate_model(model_dec, val_loader, tokenizer, device, max_samples)
        results[f"decoder_{depth}"] = {
            "params": model_dec.count_parameters(),
            **metrics_dec,
        }
        print(f"  Decoder {depth} layers CER: {metrics_dec['CER']:.4f}")

    # Efficiency metrics for full model
    model_d.eval()
    sample_images = next(iter(val_loader))["image"][:4].to(device)
    start = time.time()
    with torch.no_grad():
        for img in sample_images:
            _ = model_d.generate(img.unsqueeze(0), max_new_tokens=16, temperature=0.01)
    gen_time = (time.time() - start) / 4  # per sample

    # Measure model file size
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    torch.save(model_d.state_dict(), tmp_path)
    model_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    results["_efficiency"] = {
        "total_params": model_d.count_parameters(),
        "model_size_mb": model_size_mb,
        "avg_gen_time_per_sample_s": gen_time,
        "tokens_per_second": 16 / gen_time if gen_time > 0 else 0,
        "gpu_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0,
    }

    print(f"\n  Efficiency: {model_d.count_parameters():,} params, {model_size_mb:.1f}MB model")
    print(f"  Generation time: {gen_time:.3f}s/sample")

    return results


def print_ablation_results(results: dict):
    """Pretty-print ablation study results."""
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)

    # Summary table
    print(f"\n{'Model':<25} {'Params':>12} {'CER':>8} {'WER':>8} {'EM':>8}")
    print("-"*65)

    for name, metrics in results.items():
        if name.startswith("_"):
            continue
        params_k = metrics.get("params", 0) / 1e6
        cer = metrics.get("CER", 0)
        wer = metrics.get("WER", 0)
        em = metrics.get("ExactMatch", 0)
        print(f"{name:<25} {params_k:>10.1f}M {cer:>8.4f} {wer:>8.4f} {em:>8.4f}")

    print("\nEfficiency Metrics:")
    if "_efficiency" in results:
        eff = results["_efficiency"]
        for k, v in eff.items():
            print(f"  {k}: {v}")
    print("="*80)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(PROJECT_ROOT, "data", "synthetic_data")
    dataloaders = create_train_val_test_loaders(data_path, batch_size=16)
    tokenizer = OCRTokenizer(vocab_size=2000)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

    if os.path.exists(checkpoint_path):
        model, config = load_model(checkpoint_path, device)
        metrics = evaluate_model(model, dataloaders["test"], tokenizer, device)
        print("\nEvaluation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    else:
        print("No checkpoint found. Run training first.")
        print("\nRunning ablation study...")
        results = run_ablation_study(data_path)
        print_ablation_results(results)
