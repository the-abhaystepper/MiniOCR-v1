"""HuggingFace Hub Upload Script for MiniOCR Synthetic Dataset.

Usage:
    python upload_to_hf.py --repo-id your-username/miniocr-synthetic-ocr
"""

import os
import json
import argparse
from glob import glob
from datasets import Dataset, DatasetDict, Image as DImage


def prepare_hf_dataset(data_dir: str = "data/synthetic_data"):
    """Convert on-disk synthetic dataset to Hugging Face DatasetDict."""
    categories = ["easy", "medium", "hard", "receipt", "form", "table", "document"]

    samples = []
    for cat in categories:
        cat_dir = os.path.join(data_dir, cat)
        if not os.path.isdir(cat_dir):
            continue

        for png_path in sorted(glob(os.path.join(cat_dir, "*.png"))):
            json_path = png_path.replace(".png", ".json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                samples.append({
                    "image": png_path,
                    "text": meta["text"],
                    "difficulty": meta.get("difficulty", cat),
                    "doc_type": meta.get("doc_type", cat),
                    "bounding_boxes": str(meta.get("bounding_boxes", [])),
                    "reading_order": str(meta.get("reading_order", [])),
                })

    # Create HF dataset
    ds = Dataset.from_list(samples)
    ds = ds.cast_column("image", DImage())

    # Train / Val / Test split (80 / 10 / 10)
    splits = ds.train_test_split(test_size=0.2, seed=42)
    val_test = splits["test"].train_test_split(test_size=0.5, seed=42)

    dataset_dict = DatasetDict({
        "train": splits["train"],
        "validation": val_test["train"],
        "test": val_test["test"],
    })

    return dataset_dict


def main():
    parser = argparse.ArgumentParser(description="Upload synthetic OCR dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID (e.g. username/miniocr-synthetic)")
    parser.add_argument("--data-dir", type=str, default="data/synthetic_data", help="Path to synthetic data directory")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    print(f"Building Hugging Face dataset from {args.data_dir}...")
    dataset_dict = prepare_hf_dataset(args.data_dir)
    print(dataset_dict)

    print(f"Pushing dataset to Hugging Face Hub: {args.repo_id}...")
    dataset_dict.push_to_hub(args.repo_id, private=args.private)
    print("Upload complete!")


if __name__ == "__main__":
    main()
