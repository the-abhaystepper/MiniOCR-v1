"""Evaluation and ablation study script for MiniOCR."""

import os
import sys
import tempfile
import time
import torch
import torch.nn as nn
from torch.amp import autocast

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.miniocr import MiniOCR
from models.metrics import batch_metrics
from tokenizer.tokenizer import OCRTokenizer
from data.dataset import create_train_val_test_loaders


def load_model(checkpoint_path: str, device: torch.device = None) -> tuple[MiniOCR, dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MiniOCR(
        vocab_size=2000, image_size=224, patch_size=16,
        vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
        qformer_queries=96, qformer_dim=384, qformer_depth=3, qformer_heads=6,
        qformer_ffn=1024, decoder_dim=384, decoder_depth=8, decoder_heads=6,
        decoder_ffn=1536, max_seq_len=256,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate_model(model, dataloader, tokenizer, device, max_samples=200) -> dict:
    model.eval()
    predictions, targets, total_loss = [], [], 0.0
    pad_id = tokenizer.pad_token_id
    use_amp = device.type == "cuda"

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx * dataloader.batch_size >= max_samples:
            break
        images = batch["image"].to(device)
        texts = batch["text"]

        input_ids_list = [tokenizer.encode(t, add_bos=True, add_eos=False) for t in texts]
        max_len = max(len(ids) for ids in input_ids_list)
        padded_input = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
        padded_labels = [
            ids[1:] + [tokenizer.eos_token_id] + [pad_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]
        input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, input_ids)
            total_loss += nn.CrossEntropyLoss(ignore_index=pad_id)(
                logits.reshape(-1, tokenizer.vocab_size), labels.reshape(-1)
            ).item() * images.shape[0]

        for i in range(images.shape[0]):
            gen = model.generate(images[i:i+1], max_new_tokens=64, temperature=0.01)[0]
            predictions.append(tokenizer.decode(gen.tolist()))
            targets.append(texts[i])

    metrics = batch_metrics(predictions, targets)
    metrics["loss"] = total_loss / max(len(predictions), 1)
    return metrics


@torch.no_grad()
def run_ablation_study(data_dir: str = "data/synthetic_data", batch_size: int = 16,
                       max_samples: int = 200) -> dict:
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(PROJECT_ROOT, data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader = create_train_val_test_loaders(data_dir, batch_size=batch_size)["val"]
    tokenizer = OCRTokenizer(vocab_size=2000)
    results = {}

    config_kwargs = dict(
        vocab_size=2000, image_size=224, patch_size=16,
        vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
        qformer_dim=384, qformer_depth=3, qformer_heads=6, qformer_ffn=1024,
        decoder_dim=384, decoder_heads=6, decoder_ffn=1536, max_seq_len=256,
    )

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False) if os.path.exists(checkpoint_path) else None

    # Full model
    model_d = MiniOCR(**config_kwargs, qformer_queries=96, decoder_depth=8).to(device)
    if ckpt:
        model_d.load_state_dict(ckpt["model_state_dict"])
    m = evaluate_model(model_d, val_loader, tokenizer, device, max_samples)
    results["D_full"] = {"params": model_d.count_parameters(), **m}

    # Query count ablation
    for n_queries in [32, 64, 128]:
        m = evaluate_model(MiniOCR(**config_kwargs, qformer_queries=n_queries, decoder_depth=8).to(device),
                           val_loader, tokenizer, device, max_samples)
        results[f"queries_{n_queries}"] = {"params": m["num_predictions"] and 0 or 0,
                                           **{k: v for k, v in m.items() if k != "num_predictions"}}
        results[f"queries_{n_queries}"]["params"] = MiniOCR(**config_kwargs, qformer_queries=n_queries, decoder_depth=8).to(device).count_parameters()

    # Decoder depth ablation
    for depth in [4, 6, 8]:
        m = evaluate_model(MiniOCR(**config_kwargs, qformer_queries=96, decoder_depth=depth).to(device),
                           val_loader, tokenizer, device, max_samples)
        results[f"decoder_{depth}"] = {"params": MiniOCR(**config_kwargs, qformer_queries=96, decoder_depth=depth).to(device).count_parameters(), **m}

    # Efficiency profiling
    sample_images = next(iter(val_loader))["image"][:4].to(device)
    t0 = time.time()
    with torch.no_grad():
        for img in sample_images:
            model_d.generate(img.unsqueeze(0), max_new_tokens=16, temperature=0.01)
    gen_time = (time.time() - t0) / 4

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        torch.save(model_d.state_dict(), tmp.name)
        model_size_mb = os.path.getsize(tmp.name) / (1024 * 1024)
        os.remove(tmp.name)

    results["_efficiency"] = {
        "total_params": model_d.count_parameters(),
        "model_size_mb": model_size_mb,
        "gen_time_per_sample_s": gen_time,
        "tokens_per_second": 16 / gen_time if gen_time > 0 else 0,
    }

    return results


def print_ablation_results(results: dict):
    print(f"\n{'Model':<25} {'Params':>12} {'CER':>8} {'WER':>8} {'EM':>8}")
    print("-" * 65)
    for name, m in results.items():
        if name.startswith("_"):
            continue
        print(f"{name:<25} {m.get('params',0)/1e6:>10.1f}M {m.get('CER',0):>8.4f} {m.get('WER',0):>8.4f} {m.get('ExactMatch',0):>8.4f}")
    if "_efficiency" in results:
        print("\nEfficiency:")
        for k, v in results["_efficiency"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(PROJECT_ROOT, "data", "synthetic_data")
    dataloaders = create_train_val_test_loaders(data_path, batch_size=16)
    tokenizer = OCRTokenizer(vocab_size=2000)
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

    if os.path.exists(checkpoint_path):
        model, _ = load_model(checkpoint_path, device)
        print("Test set results:")
        for k, v in evaluate_model(model, dataloaders["test"], tokenizer, device).items():
            print(f"  {k}: {v}")
    else:
        results = run_ablation_study(data_path)
        print_ablation_results(results)
