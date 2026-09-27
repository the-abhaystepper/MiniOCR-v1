"""Training script for MiniOCR on synthetic OCR dataset."""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.miniocr import MiniOCR
from models.metrics import batch_metrics, character_error_rate
from tokenizer.tokenizer import OCRTokenizer
from data.dataset import create_train_val_test_loaders


def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pad_token_id: int = 0) -> torch.Tensor:
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="mean")
    return loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))


def train_one_epoch(
    model, dataloader, tokenizer, optimizer, scheduler, scaler,
    device, epoch, use_amp=True, lambda_order=0.0,
) -> dict:
    model.train()
    total_loss = 0.0
    total_cer = 0.0
    total_samples = 0
    order_criterion = nn.MSELoss() if lambda_order > 0 else None

    for batch_idx, batch in enumerate(dataloader):
        images = batch["image"].to(device)
        texts = batch["text"]

        input_ids_list = [tokenizer.encode(t, add_bos=True, add_eos=False) for t in texts]
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id

        padded_input = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
        padded_labels = [
            ids[1:] + [tokenizer.eos_token_id] + [pad_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]

        input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, input_ids)
            loss = compute_loss(logits, labels, pad_id)

            if lambda_order > 0 and order_criterion is not None:
                _, order_scores, _ = model.encode_image(images)
                B, N = order_scores.shape
                target_order = torch.linspace(0, 1, N, device=device).unsqueeze(0).expand(B, N)
                loss = loss + lambda_order * order_criterion(order_scores, target_order)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler:
            scheduler.step()

        if batch_idx % 10 == 0:
            with torch.no_grad():
                for i in range(min(2, images.shape[0])):
                    gen = model.generate(images[i:i+1], max_new_tokens=32, temperature=0.01)[0]
                    total_cer += character_error_rate(tokenizer.decode(gen.tolist()), texts[i])
                    total_samples += 1

        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]

    return {"loss": total_loss / max(total_samples, 1), "cer": total_cer / max(total_samples, 1)}


@torch.no_grad()
def evaluate(model, dataloader, tokenizer, device, max_samples=50, use_amp=True) -> dict:
    model.eval()
    predictions, targets, total_loss, num_batches = [], [], 0.0, 0
    pad_id = tokenizer.pad_token_id

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx * dataloader.batch_size >= max_samples:
            break
        images = batch["image"].to(device)
        texts = batch["text"]

        input_ids_list = [tokenizer.encode(t, add_bos=True, add_eos=False) for t in texts]
        max_len = max(len(ids) for ids in input_ids_list)
        pad_id = tokenizer.pad_token_id

        padded_input = [ids + [pad_id] * (max_len - len(ids)) for ids in input_ids_list]
        padded_labels = [
            ids[1:] + [tokenizer.eos_token_id] + [pad_id] * (max_len - len(ids))
            for ids in input_ids_list
        ]

        input_ids = torch.tensor(padded_input, dtype=torch.long, device=device)
        labels = torch.tensor(padded_labels, dtype=torch.long, device=device)

        with autocast(device_type=device.type, enabled=use_amp, dtype=torch.bfloat16):
            logits = model(images, input_ids)
            total_loss += compute_loss(logits, labels, pad_id).item() * images.shape[0]

        for i in range(images.shape[0]):
            gen = model.generate(images[i:i+1], max_new_tokens=64, temperature=0.01)[0]
            predictions.append(tokenizer.decode(gen.tolist()))
            targets.append(texts[i])

        num_batches += 1

    metrics = batch_metrics(predictions, targets)
    metrics["loss"] = total_loss / max(len(predictions), 1)
    return metrics


def run_training(config: dict = None):
    if config is None:
        config = {
            "epochs": 50, "batch_size": 16, "learning_rate": 0.0002,
            "weight_decay": 0.01, "lambda_order": 0.1, "use_amp": True,
            "data_dir": os.path.join(PROJECT_ROOT, "data", "synthetic_data"),
            "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints"),
            "eval_interval": 5, "save_interval": 5, "patience": 10,
        }

    for key in ("data_dir", "checkpoint_dir"):
        if not os.path.isabs(config[key]):
            config[key] = os.path.join(PROJECT_ROOT, config[key])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MiniOCR(
        vocab_size=2000, image_size=224, patch_size=16,
        vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
        qformer_queries=96, qformer_dim=384, qformer_depth=3, qformer_heads=6,
        qformer_ffn=1024, decoder_dim=384, decoder_depth=8, decoder_heads=6,
        decoder_ffn=1536, max_seq_len=256,
    ).to(device)

    total_params = model.count_parameters()
    param_breakdown = model.count_parameters_by_module()
    print(f"Parameters: {total_params:,}")
    for name, count in sorted(param_breakdown.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name}: {count:,}")

    tokenizer = OCRTokenizer(vocab_size=2000)
    data_dir = config["data_dir"]

    if not os.path.exists(os.path.join(data_dir, "easy")):
        from data.synthetic_data.generator import SyntheticOCRDataset
        SyntheticOCRDataset(output_dir=data_dir).build_dataset(
            n_easy=300, n_medium=300, n_hard=300, n_document=300
        )

    dataloaders = create_train_val_test_loaders(data_dir=data_dir, batch_size=int(config["batch_size"]))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["learning_rate"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    scaler = GradScaler(enabled=config.get("use_amp", True) and device.type == "cuda")

    best_cer, patience_counter = float('inf'), 0
    use_amp = config.get("use_amp", True) and device.type == "cuda"

    for epoch in range(int(config["epochs"])):
        t0 = time.time()
        train_m = train_one_epoch(
            model, dataloaders["train"], tokenizer, optimizer, scheduler,
            scaler, device, epoch, use_amp=use_amp,
            lambda_order=float(config.get("lambda_order", 0.0)),
        )

        if (epoch + 1) % int(config.get("eval_interval", 5)) == 0 or epoch == 0:
            val_m = evaluate(model, dataloaders["val"], tokenizer, device,
                             max_samples=100, use_amp=use_amp)
            print(
                f"Epoch {epoch+1}/{config['epochs']} | "
                f"Train Loss: {train_m['loss']:.4f} | "
                f"Val Loss: {val_m['loss']:.4f} | "
                f"Val CER: {val_m['CER']:.4f}"
            )
            if val_m["CER"] < best_cer:
                best_cer = val_m["CER"]
                patience_counter = 0
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch, "val_cer": best_cer,
                }, os.path.join(config["checkpoint_dir"], "best_model.pt"))
            else:
                patience_counter += 1
            if patience_counter >= int(config.get("patience", 10)):
                print(f"Early stopping at epoch {epoch+1}")
                break
        else:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_m['loss']:.4f} ({time.time()-t0:.1f}s)")

        if (epoch + 1) % int(config.get("save_interval", 5)) == 0:
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
            }, os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pt"))

    print(f"Training complete. Best Val CER: {best_cer:.4f}")

    with open(os.path.join(config["checkpoint_dir"], "model_config.json"), "w") as f:
        json.dump({
            "total_parameters": total_params,
            "parameter_breakdown": param_breakdown,
            "best_val_cer": best_cer,
            "training_config": config,
        }, f, indent=2)

    return model


if __name__ == "__main__":
    import yaml
    config_file = os.path.join(PROJECT_ROOT, "config", "base.yaml")
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        training_config = {
            **{k: (float(v) if isinstance(v, str) and v.replace(".", "", 1).lstrip("-").isdigit() else v)
               for k, v in yaml_config.get("training", {}).items()},
            **{
                "data_dir": os.path.join(PROJECT_ROOT, yaml_config["data"]["data_dir"]),
                "checkpoint_dir": os.path.join(PROJECT_ROOT, yaml_config["logging"]["checkpoint_dir"]),
            },
        }
    else:
        training_config = None
    run_training(training_config)
