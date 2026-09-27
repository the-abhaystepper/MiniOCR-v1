"""Training script for MiniOCR — pretraining on synthetic OCR dataset.

Features:
    - Teacher-forcing with shifted targets
    - AdamW optimizer with warmup
    - Mixed precision training (AMP)
    - Gradient clipping
    - Early stopping
    - Checkpointing
    - Order-loss auxiliary training option
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Project imports
from models.miniocr import MiniOCR
from models.metrics import batch_metrics, character_error_rate, word_error_rate
from tokenizer.tokenizer import OCRTokenizer
from data.dataset import OCRImageDataset, create_train_val_test_loaders


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """Cross-entropy loss with padding ignored.

    Args:
        logits: [B, T, vocab_size] model output logits
        labels: [B, T] target token IDs
        pad_token_id: index of pad token to ignore
    Returns:
        Scalar loss tensor
    """
    B, T, V = logits.shape
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="mean")
    loss = loss_fn(logits.reshape(-1, V), labels.reshape(-1))
    return loss


def train_one_epoch(
    model: MiniOCR,
    dataloader: DataLoader,
    tokenizer: OCRTokenizer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    lambda_order: float = 0.0,
) -> dict:
    """Train for one epoch.

    Args:
        model: MiniOCR model
        dataloader: Training DataLoader
        tokenizer: OCR tokenizer
        optimizer: AdamW optimizer
        scheduler: LR scheduler (optional)
        scaler: PyTorch GradScaler for AMP
        device: torch.device
        epoch: Current epoch number
        use_amp: Use mixed precision
        lambda_order: Weight for order auxiliary loss
    Returns:
        dict with training metrics
    """
    model.train()
    total_loss = 0.0
    total_cer = 0.0
    total_samples = 0
    num_batches = 0

    # For order loss
    order_criterion = nn.MSELoss() if lambda_order > 0 else None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)          # [B, 3, 224, 224]
        texts = batch["text"]                         # list of strings

        # Tokenize texts
        input_ids_list = []
        for text in texts:
            # Encode with BOS
            encoded = tokenizer.encode(text, add_bos=True, add_eos=False)
            input_ids_list.append(encoded)

        # Pad input_ids and create labels
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id

        padded_input = []
        padded_labels = []
        for ids in input_ids_list:
            # Input: BOS + tokens
            inp = ids + [pad_id] * (max_len - len(ids))
            # Labels: tokens + EOS + pad (shifted)
            lbl = ids[1:] + [tokenizer.eos_token_id] + [pad_id] * (max_len - len(ids))
            padded_input.append(inp)
            padded_labels.append(lbl)

        input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)   # [B, T]
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)      # [B, T]

        # Teacher forcing: forward pass
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, input_ids)  # [B, T, vocab_size]

            # Main OCR loss
            loss = compute_loss(logits, labels, pad_token_id=pad_id)

            # Optional order auxiliary loss
            if lambda_order > 0 and order_criterion is not None:
                # Get order scores from model
                ocr_tokens, order_scores, _ = model.encode_image(images)
                # Target: normalized position in sequence for each token
                B, N = order_scores.shape  # [B, 96]
                target_order = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, N)
                order_loss = order_criterion(order_scores, target_order)
                loss = loss + lambda_order * order_loss

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        # Compute CER on decoded predictions (sample a few)
        if batch_idx % 10 == 0:
            with torch.no_grad():
                for i in range(min(2, images.shape[0])):
                    generated_ids = model.generate(
                        images[i:i+1], max_new_tokens=32, temperature=0.01
                    )[0]
                    pred_text = tokenizer.decode(generated_ids.tolist())
                    true_text = texts[i]
                    total_cer += character_error_rate(pred_text, true_text)
                    total_samples += 1

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
        num_batches += 1

    avg_loss = total_loss / max(total_samples, 1)
    avg_cer = total_cer / max(total_samples, 1)

    return {
        "loss": avg_loss,
        "cer": avg_cer,
        "batches": num_batches,
    }


@torch.no_grad()
def evaluate(
    model: MiniOCR,
    dataloader: DataLoader,
    tokenizer: OCRTokenizer,
    device: torch.device,
    max_samples: int = 50,
    use_amp: bool = True,
) -> dict:
    """Evaluate model on validation/test set.

    Args:
        model: MiniOCR model
        dataloader: DataLoader
        tokenizer: OCR tokenizer
        device: torch.device
        max_samples: Max samples to evaluate
        use_amp: Use mixed precision
    Returns:
        dict with evaluation metrics
    """
    model.eval()
    predictions = []
    targets = []
    total_loss = 0.0
    num_batches = 0
    pad_id = tokenizer.pad_token_id

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
            inp = ids + [pad_id] * (max_len - len(ids))
            lbl = ids[1:] + [tokenizer.eos_token_id] + [pad_id] * (max_len - len(ids))
            padded_input.append(inp)
            padded_labels.append(lbl)

        input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

        # Forward pass
        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, input_ids)
            loss = compute_loss(logits, labels, pad_token_id=pad_id)
        total_loss += loss.item() * images.shape[0]
        num_batches += 1

        # Generate predictions
        for i in range(images.shape[0]):
            generated_ids = model.generate(
                images[i:i+1], max_new_tokens=64, temperature=0.01
            )[0]
            pred_text = tokenizer.decode(generated_ids.tolist())
            predictions.append(pred_text)
            targets.append(texts[i])

    avg_loss = total_loss / max(num_batches * dataloader.batch_size, 1)
    metrics = batch_metrics(predictions, targets)
    metrics["loss"] = avg_loss

    return metrics


def run_training(config: dict = None):
    """Main training pipeline.

    Args:
        config: Dictionary with training configuration
    """
    if config is None:
        config = {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.0002,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "grad_clip": 1.0,
            "lambda_order": 0.1,
            "use_amp": True,
            "data_dir": os.path.join(PROJECT_ROOT, "data", "synthetic_data"),
            "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
            "eval_interval": 5,
            "save_interval": 5,
        }

    # Ensure paths are absolute or resolved relative to project root
    if not os.path.isabs(config["data_dir"]):
        config["data_dir"] = os.path.join(PROJECT_ROOT, config["data_dir"])
    if not os.path.isabs(config["checkpoint_dir"]):
        config["checkpoint_dir"] = os.path.join(PROJECT_ROOT, config["checkpoint_dir"])

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Build model ──
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

    total_params = model.count_parameters()
    param_breakdown = model.count_parameters_by_module()

    print(f"\n{'='*60}")
    print(f"MiniOCR Architecture Summary")
    print(f"{'='*60}")
    print(f"Total trainable parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"\nParameter breakdown by module:")
    for name, count in sorted(param_breakdown.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"  {name}: {count:,}")
    print(f"{'='*60}\n")

    model.to(device)

    # ── Build tokenizer ──
    tokenizer = OCRTokenizer(vocab_size=2000)
    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # ── Build dataset ──
    data_dir = config["data_dir"]
    if not os.path.exists(os.path.join(data_dir, "easy")):
        print("Synthetic data not found. Generating dataset...")
        from data.synthetic_data.generator import SyntheticOCRDataset
        dataset_builder = SyntheticOCRDataset(output_dir=data_dir)
        dataset_builder.build_dataset(
            n_easy=200, n_medium=150, n_hard=100, n_document=100
        )

    # ── DataLoaders ──
    dataloaders = create_train_val_test_loaders(data_dir=data_dir, batch_size=int(config["batch_size"]))
    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    # ── Optimizer and scheduler ──
    learning_rate = float(config["learning_rate"])
    weight_decay = float(config.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(float(config.get("beta1", 0.9)), float(config.get("beta2", 0.999))),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # ── Training loop ──
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    scaler = GradScaler(enabled=config.get("use_amp", True) and device.type == "cuda")

    best_cer = float('inf')
    patience_counter = 0

    print(f"\nStarting training for {config['epochs']} epochs...")
    print(f"Batch size: {config['batch_size']}, LR: {config['learning_rate']}")
    print(f"Lambda order: {config.get('lambda_order', 0.0)}")

    for epoch in range(int(config["epochs"])):
        t0 = time.time()

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, tokenizer, optimizer, scheduler,
            scaler, device, epoch,
            use_amp=config.get("use_amp", True) and device.type == "cuda",
            lambda_order=float(config.get("lambda_order", 0.0)),
        )

        # Evaluate
        if (epoch + 1) % int(config.get("eval_interval", 5)) == 0 or epoch == 0:
            val_metrics = evaluate(
                model, val_loader, tokenizer, device,
                max_samples=100,
                use_amp=config.get("use_amp", True) and device.type == "cuda",
            )

            epoch_time = time.time() - t0
            print(
                f"Epoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Val CER: {val_metrics['CER']:.4f} | "
                f"Val WER: {val_metrics['WER']:.4f} | "
                f"EM: {val_metrics['ExactMatch']:.3f}"
            )

            # Save best model
            if val_metrics["CER"] < best_cer:
                best_cer = val_metrics["CER"]
                patience_counter = 0
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_cer": val_metrics["CER"],
                    "val_wer": val_metrics["WER"],
                    "total_params": total_params,
                }
                torch.save(checkpoint, os.path.join(config["checkpoint_dir"], "best_model.pt"))
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= int(config.get("patience", 10)):
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            epoch_time = time.time() - t0
            print(
                f"Epoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s) | "
                f"Train Loss: {train_metrics['loss']:.4f}"
            )

        # Save periodic checkpoint
        if (epoch + 1) % int(config.get("save_interval", 5)) == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pt"))

    print("\nTraining complete!")
    print(f"Best validation CER: {best_cer:.4f}")

    # Save final model config
    final_config = {
        "model_config": {
            "vocab_size": 2000,
            "image_size": 224,
            "patch_size": 16,
            "vision_dim": 256,
            "vision_depth": 6,
            "vision_heads": 4,
            "vision_ffn": 512,
            "qformer_queries": 96,
            "qformer_dim": 384,
            "qformer_depth": 3,
            "qformer_heads": 6,
            "qformer_ffn": 1024,
            "decoder_dim": 384,
            "decoder_depth": 8,
            "decoder_heads": 6,
            "decoder_ffn": 1536,
            "max_seq_len": 256,
        },
        "total_parameters": total_params,
        "parameter_breakdown": param_breakdown,
        "best_val_cer": best_cer,
        "training_config": config,
    }
    with open(os.path.join(config["checkpoint_dir"], "model_config.json"), "w") as f:
        json.dump(final_config, f, indent=2)

    return model


if __name__ == "__main__":
    import yaml
    config_file = os.path.join(PROJECT_ROOT, "config", "base.yaml")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        training_config = {
            **{k: float(v) if isinstance(v, str) and v.replace(".", "", 1).lstrip("-").isdigit() else v for k, v in yaml_config.get("training", {}).items()},
            **{
                "data_dir": os.path.join(PROJECT_ROOT, yaml_config["data"]["data_dir"]),
                "checkpoint_dir": os.path.join(PROJECT_ROOT, yaml_config["logging"]["checkpoint_dir"]),
            },
        }
    else:
        training_config = None

    run_training(training_config)
