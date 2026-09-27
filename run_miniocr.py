"""MiniOCR — Main entry point.

Usage:
    python run_miniocr.py train        # Train the model
    python run_miniocr.py generate     # Generate synthetic data
    python run_miniocr.py eval         # Evaluate trained model
    python run_miniocr.py ablation     # Run ablation studies
    python run_miniocr.py test         # Test everything end-to-end

Example:
    python run_miniocr.py test
"""

import os
import sys
import json
import torch

# Set UTF-8 encoding for stdout/stderr if possible
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def generate_data():
    """Generate synthetic OCR dataset."""
    print("=" * 60)
    print("STEP 1: Generating Synthetic OCR Dataset")
    print("=" * 60)
    from data.synthetic_data.generator import SyntheticOCRDataset
    dataset_builder = SyntheticOCRDataset(output_dir=os.path.join(PROJECT_ROOT, "data", "synthetic_data"))
    samples = dataset_builder.build_dataset(
        n_easy=200, n_medium=150, n_hard=100, n_document=100
    )
    stats = dataset_builder.get_statistics()
    print(f"\nDataset statistics: {stats}")
    print(f"Total samples: {len(samples)}")
    print(f"Dataset saved to data/synthetic_data/")
    return samples


def train():
    """Train the MiniOCR model."""
    print("=" * 60)
    print("STEP 2: Training MiniOCR")
    print("=" * 60)
    from training.train import run_training
    import yaml

    config_path = os.path.join(PROJECT_ROOT, "config", "base.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)

        training_config = {
            **yaml_config.get("training", {}),
            **{
                "data_dir": os.path.join(PROJECT_ROOT, yaml_config["data"]["data_dir"]),
                "checkpoint_dir": os.path.join(PROJECT_ROOT, yaml_config["logging"]["checkpoint_dir"]),
            },
        }
    else:
        training_config = None

    model = run_training(training_config)
    return model


def evaluate():
    """Evaluate trained MiniOCR model."""
    print("=" * 60)
    print("STEP 3: Evaluating MiniOCR")
    print("=" * 60)
    from training.evaluate import evaluate_model, load_model, run_ablation_study, print_ablation_results
    from data.dataset import create_train_val_test_loaders
    from tokenizer.tokenizer import OCRTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = os.path.join(PROJECT_ROOT, "data", "synthetic_data")
    dataloaders = create_train_val_test_loaders(data_dir, batch_size=16)
    tokenizer = OCRTokenizer(vocab_size=2000)

    checkpoint_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_model.pt")

    if os.path.exists(checkpoint_path):
        model, config = load_model(checkpoint_path, device)
        print(f"\nLoaded model from {checkpoint_path}")
        print(f"Total parameters: {config.get('total_params', 'N/A')}")

        # Evaluate on test set
        print("\nEvaluating on test set...")
        metrics = evaluate_model(model, dataloaders["test"], tokenizer, device, max_samples=100)
        print("\nTest Results:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")

        # Run ablation studies
        print("\nRunning ablation studies...")
        results = run_ablation_study(data_dir, batch_size=16, max_samples=100)
        print_ablation_results(results)
    else:
        print("No checkpoint found. Run training first.")
        print("\nRunning ablation studies on untrained model...")
        results = run_ablation_study(data_dir, batch_size=16, max_samples=50)
        print_ablation_results(results)


def test_architecture():
    """Test the complete architecture with a forward pass."""
    print("=" * 60)
    print("STEP 0: Architecture Validation")
    print("=" * 60)

    from models.miniocr import MiniOCR

    # Build the model
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
            pct = count / total_params * 100
            print(f"  {name}: {count:,} ({pct:.1f}%)")
    print(f"{'='*60}\n")

    # Test forward pass with dummy data
    print("Testing forward pass...")
    model.eval()
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 2000, (2, 20))

    with torch.no_grad():
        logits = model(dummy_images, dummy_input_ids)

    print(f"  Input images:       {dummy_images.shape}")
    print(f"  Input token IDs:    {dummy_input_ids.shape}")
    print(f"  Output logits:      {logits.shape}")
    print(f"  Expected:           [2, 20, 2000]")
    assert logits.shape == (2, 20, 2000), f"Shape mismatch: {logits.shape}"
    print("  [OK] Forward pass successful!\n")

    # Test image encoding
    print("Testing image encoding...")
    with torch.no_grad():
        ocr_tokens, order_scores, visual_features = model.encode_image(dummy_images)
    print(f"  Visual features:    {visual_features.shape}")
    print(f"  OCR tokens:         {ocr_tokens.shape}")
    print(f"  Order scores:       {order_scores.shape}")
    assert visual_features.shape == (2, 196, 256), f"Visual features shape wrong: {visual_features.shape}"
    assert ocr_tokens.shape == (2, 96, 384), f"OCR tokens shape wrong: {ocr_tokens.shape}"
    assert order_scores.shape == (2, 96), f"Order scores shape wrong: {order_scores.shape}"
    print("  [OK] Image encoding successful!\n")

    # Test generation
    print("Testing autoregressive generation...")
    with torch.no_grad():
        generated = model.generate(dummy_images[:1], max_new_tokens=16, temperature=0.01)
    print(f"  Generated tokens:   {generated.shape}")
    print(f"  Generated IDs:      {generated[0].tolist()}")
    assert generated.shape == (1, 17), f"Generated shape wrong: {generated.shape}"  # BOS + 16
    print("  [OK] Generation successful!\n")

    # Check for NaN
    has_nan = torch.isnan(logits).any()
    print(f"  NaN check:          {'FAIL' if has_nan else 'PASS'}")

    # Check gradient flow
    print("\nTesting backward pass...")
    model.train()
    train_logits = model(dummy_images, dummy_input_ids)
    loss = torch.nn.functional.cross_entropy(train_logits.reshape(-1, 2000), dummy_input_ids.reshape(-1), ignore_index=0)
    loss.backward()
    has_grad = any(p.grad is not None and not torch.isnan(p.grad).any() for p in model.parameters() if p.requires_grad)
    print(f"  Gradient flow:      {'PASS' if has_grad else 'FAIL'}")
    print(f"  Loss:               {loss.item():.4f}")

    print("\n" + "="*60)
    print("ALL ARCHITECTURE TESTS PASSED!")
    print("="*60)

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MiniOCR — VLM-based OCR")
    parser.add_argument("command", choices=["train", "generate", "eval", "ablation", "test", "arch", "all"],
                        help="Command to run")
    args = parser.parse_args()

    if args.command == "arch":
        test_architecture()
    elif args.command == "test":
        # Test architecture first
        test_architecture()
        # Generate data
        generate_data()
        # Train
        train()
        # Evaluate
        evaluate()
    elif args.command == "generate":
        generate_data()
    elif args.command == "train":
        train()
    elif args.command == "eval":
        evaluate()
    elif args.command == "ablation":
        from training.evaluate import run_ablation_study, print_ablation_results
        results = run_ablation_study(os.path.join(PROJECT_ROOT, "data", "synthetic_data"), batch_size=16, max_samples=50)
        print_ablation_results(results)
    elif args.command == "all":
        test_architecture()
        generate_data()
        train()
        evaluate()
