"""MiniOCR entry point: train, generate data, evaluate, or test."""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def generate_data():
    from data.synthetic_data.generator import SyntheticOCRDataset
    ds = SyntheticOCRDataset(output_dir=os.path.join(PROJECT_ROOT, "data", "synthetic_data"))
    ds.build_dataset(n_easy=300, n_medium=300, n_hard=300, n_document=300)
    print(f"Generated {len(ds.samples)} samples")


def train():
    from training.train import run_training
    import yaml
    config_path = os.path.join(PROJECT_ROOT, "config", "base.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        training_config = {
            **{k: (float(v) if isinstance(v, str) and v.replace(".", "", 1).lstrip("-").isdigit() else v)
               for k, v in cfg.get("training", {}).items()},
            **{
                "data_dir": os.path.join(PROJECT_ROOT, cfg["data"]["data_dir"]),
                "checkpoint_dir": os.path.join(PROJECT_ROOT, cfg["logging"]["checkpoint_dir"]),
            },
        }
    else:
        training_config = None
    run_training(training_config)


def evaluate():
    from training.evaluate import evaluate_model, load_model, run_ablation_study, print_ablation_results
    from data.dataset import create_train_val_test_loaders
    from tokenizer.tokenizer import OCRTokenizer
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_data")
    dataloaders = create_train_val_test_loaders(data_dir, batch_size=16)
    tokenizer = OCRTokenizer(vocab_size=2000)
    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

    if os.path.exists(checkpoint_path):
        model, _ = load_model(checkpoint_path, device)
        print("Test set:")
        for k, v in evaluate_model(model, dataloaders["test"], tokenizer, device).items():
            print(f"  {k}: {v}")
        print("\nAblation:")
        print_ablation_results(run_ablation_study(data_dir))
    else:
        print("No checkpoint found. Run training first.")
        results = run_ablation_study(data_dir)
        print_ablation_results(results)


def test_architecture():
    import torch
    from models.miniocr import MiniOCR

    model = MiniOCR(
        vocab_size=2000, image_size=224, patch_size=16,
        vision_dim=256, vision_depth=6, vision_heads=4, vision_ffn=512,
        qformer_queries=96, qformer_dim=384, qformer_depth=3, qformer_heads=6,
        qformer_ffn=1024, decoder_dim=384, decoder_depth=8, decoder_heads=6,
        decoder_ffn=1536, max_seq_len=256,
    )
    print(f"Parameters: {model.count_parameters():,}")

    model.eval()
    imgs = torch.randn(2, 3, 224, 224)
    ids = torch.randint(0, 2000, (2, 20))
    with torch.no_grad():
        logits = model(imgs, ids)
        ocr_tok, order, _ = model.encode_image(imgs)
        gen = model.generate(imgs[:1], max_new_tokens=16, temperature=0.01)

    assert logits.shape == (2, 20, 2000)
    assert ocr_tok.shape == (2, 96, 384)
    assert order.shape == (2, 96)
    assert gen.shape == (1, 17)
    print("[OK] Forward pass")

    # Recompute in train mode so gradients flow
    model.train()
    logits_train = model(imgs, ids)
    loss = torch.nn.functional.cross_entropy(logits_train.reshape(-1, 2000), ids.reshape(-1), ignore_index=0)
    loss.backward()
    has_grad = any(p.grad is not None and not torch.isnan(p.grad).any()
                   for p in model.parameters() if p.requires_grad)
    print(f"[OK] Backward pass: {has_grad}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniOCR")
    parser.add_argument("command", choices=["train", "generate", "eval", "ablation", "test", "arch", "all"])
    args = parser.parse_args()

    if args.command == "arch":
        test_architecture()
    elif args.command == "test":
        test_architecture()
        generate_data()
        train()
        evaluate()
    elif args.command == "generate":
        generate_data()
    elif args.command == "train":
        train()
    elif args.command == "eval":
        evaluate()
    elif args.command == "ablation":
        from training.evaluate import run_ablation_study, print_ablation_results
        import os as _os
        print_ablation_results(run_ablation_study(_os.path.join(PROJECT_ROOT, "data", "synthetic_data")))
    elif args.command == "all":
        test_architecture()
        generate_data()
        train()
        evaluate()
