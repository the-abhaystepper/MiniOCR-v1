"""Synthetic OCR dataset generator using Pillow.

Generates realistic OCR training images with multiple difficulty levels:
    - Easy:     clean single-line, black text, standard fonts
    - Medium:   multi-line, variable fonts/sizes, alignment changes, grayscale
    - Hard:     blur, noise, compression, rotation, perspective distortion
    - Document: paragraphs, receipts, forms, tables, multi-column layouts

Each generated example includes metadata:
    - image (PIL Image)
    - text (str)
    - bounding boxes (list of [x1,y1,x2,y2])
    - line IDs (list)
    - reading order (list of float scores)
"""

import os
import json
import random
import io
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np

# Standard safe Windows fonts to try
_STANDARD_FONTS = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\times.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    r"C:\Windows\Fonts\cour.ttf",
    r"C:\Windows\Fonts\courbd.ttf",
    r"C:\Windows\Fonts\consola.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\tahoma.ttf",
    r"C:\Windows\Fonts\verdana.ttf",
]

# Keep only existing font files
_VALID_FONTS = [f for f in _STANDARD_FONTS if os.path.isfile(f)]


def _load_font(size: int, bold: bool = False, italic: bool = False):
    """Load a TrueType font safely, falling back to default."""
    if not _VALID_FONTS:
        return ImageFont.load_default()

    candidates = _VALID_FONTS
    if bold:
        bold_candidates = [f for f in candidates if 'bd' in f.lower() or 'b.ttf' in f.lower()]
        if bold_candidates:
            candidates = bold_candidates

    font_path = random.choice(candidates)
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


# ── Easy: Clean single-line / multi-line ──

def generate_easy_sample(output_size=(224, 224)) -> dict:
    """Generate a clean, easy OCR sample."""
    texts = [
        "HELLO WORLD",
        "MiniOCR 2024",
        "The quick brown fox",
        "ABC123 XYZ789",
        "Welcome to OCR",
        "Sample Text Here",
        "PYTHON ROCKS",
        "Data Science AI",
        "Neural Network",
        "Machine Learning",
        "Deep Vision Model",
        "Image Recognition",
        "Clean Document",
        "Simple Text Line",
        "OCR Benchmark",
    ]

    text = random.choice(texts)
    font_size = random.choice([20, 24, 28, 32])
    font = _load_font(font_size, bold=random.random() < 0.3)

    img_w, img_h = output_size
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    # Estimate text dimensions
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # Center text
    x = max(8, (img_w - text_w) // 2)
    y = max(8, (img_h - text_h) // 2)
    draw.text((x, y), text, fill="black", font=font)

    return {
        "image": img,
        "text": text,
        "bounding_boxes": [[x, y, x + text_w, y + text_h]],
        "line_ids": [0],
        "reading_order": [0.5],
        "difficulty": "easy",
    }


def generate_easy_multi_line(num_lines: int = None) -> dict:
    """Generate a multi-line easy OCR sample."""
    if num_lines is None:
        num_lines = random.randint(2, 4)

    lines_text = [
        "Name: John Doe",
        "Address: 123 Main St",
        "City: Springfield",
        "Zip: 62701",
        "Phone: 555-0123",
        "Email: john@example.com",
        "Date: 2024-01-15",
        "Total: $42.99",
        "Item: Widget A",
        "Qty: 3",
        "Status: Complete",
        "ID: ABC-1234",
    ]
    selected = random.sample(lines_text, min(num_lines, len(lines_text)))

    font_size = random.choice([16, 18, 20])
    font = _load_font(font_size)

    img_w, img_h = 224, 224
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    line_height = font_size + 8
    total_h = num_lines * line_height
    start_y = max(8, (img_h - total_h) // 2)

    all_bboxes = []
    all_texts = []
    all_order = []
    all_line_ids = []
    full_text = ""

    for i, line in enumerate(selected):
        y = start_y + i * line_height
        align = random.choice(["left", "center"])
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        if align == "left":
            x = 12
        else:
            x = max(12, (img_w - tw) // 2)

        draw.text((x, y), line, fill="black", font=font)
        all_bboxes.append([x, y, x + tw, y + th])
        all_texts.append(line)
        all_order.append(i / max(1, num_lines - 1))
        all_line_ids.append(i)
        full_text += line + "\n"

    return {
        "image": img,
        "text": full_text.strip(),
        "bounding_boxes": all_bboxes,
        "line_ids": all_line_ids,
        "reading_order": all_order,
        "difficulty": "easy",
    }


# ── Medium: Variable fonts, sizes, colors ──

def generate_medium_sample() -> dict:
    """Generate a medium-difficulty OCR sample."""
    num_lines = random.randint(3, 5)
    texts = [
        "Invoice #INV-2024-0042",
        "Date: March 15, 2024",
        "Customer: Acme Corp",
        "Item: Widget Pro 100",
        "Qty: 5  Price: $29.99",
        "Subtotal: $149.95",
        "Tax (8%): $12.00",
        "Grand Total: $161.95",
        "Thank you for your business!",
    ]
    selected = random.sample(texts, min(num_lines, len(texts)))

    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)

    all_bboxes = []
    all_texts = []
    all_order = []
    all_line_ids = []
    full_text = ""

    y_offset = 12
    for i, text in enumerate(selected):
        font_size = random.choice([14, 16, 18])
        font = _load_font(font_size, bold=random.random() < 0.3)

        color = (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150)) if random.random() < 0.2 else "black"
        x_offset = random.randint(8, 20)

        bbox = draw.textbbox((x_offset, y_offset), text, font=font)
        draw.text((x_offset, y_offset), text, fill=color, font=font)

        all_bboxes.append([x_offset, y_offset, bbox[2], bbox[3]])
        all_texts.append(text)
        all_order.append(i / max(1, num_lines - 1))
        all_line_ids.append(i)
        full_text += text + "\n"
        y_offset += (bbox[3] - bbox[1]) + random.randint(6, 12)

    return {
        "image": img,
        "text": full_text.strip(),
        "bounding_boxes": all_bboxes,
        "line_ids": all_line_ids,
        "reading_order": all_order,
        "difficulty": "medium",
    }


# ── Hard: Noise, blur, rotation, artifacts ──

def generate_hard_sample() -> dict:
    """Generate a hard-difficulty OCR sample with visual artifacts."""
    base = generate_easy_multi_line(num_lines=random.randint(2, 4))
    img = base["image"].copy()
    text = base["text"]

    artifact = random.choice(["blur", "noise", "contrast", "rotation", "brightness"])

    if artifact == "blur":
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.6, 1.5)))
    elif artifact == "noise":
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, random.uniform(15, 30), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    elif artifact == "contrast":
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.4, 0.7))
    elif artifact == "rotation":
        img = img.rotate(random.uniform(-10, 10), fillcolor="white")
    elif artifact == "brightness":
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.5, 0.8))

    return {
        "image": img,
        "text": text,
        "bounding_boxes": base["bounding_boxes"],
        "line_ids": base["line_ids"],
        "reading_order": base["reading_order"],
        "difficulty": "hard",
    }


# ── Document-style: receipts, forms, tables ──

def generate_document_sample(doc_type: str = None) -> dict:
    """Generate a structured document sample (receipt, form, or table)."""
    if doc_type is None:
        doc_type = random.choice(["receipt", "form", "table"])

    img = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(img)

    all_bboxes = []
    all_texts = []
    all_order = []
    all_line_ids = []
    full_text = ""
    y_offset = 10

    if doc_type == "receipt":
        items = [
            ("COFFEE SHOP", 16, True),
            ("------------------------", 12, False),
            ("Coffee         $3.50", 12, False),
            ("Sandwich       $8.99", 12, False),
            ("Water          $1.50", 12, False),
            ("------------------------", 12, False),
            ("Total:        $13.99", 14, True),
            ("Thank you!", 12, False),
        ]
        for line, fs, bold in items:
            font = _load_font(fs, bold=bold)
            bbox = draw.textbbox((10, y_offset), line, font=font)
            draw.text((10, y_offset), line, fill="black", font=font)
            all_bboxes.append([10, y_offset, bbox[2], bbox[3]])
            all_texts.append(line)
            all_order.append(len(all_texts) / len(items))
            all_line_ids.append(len(all_texts) - 1)
            full_text += line + "\n"
            y_offset += (bbox[3] - bbox[1]) + 4

    elif doc_type == "form":
        fields = [
            "Name: ________________",
            "Date: ___/___/___",
            "Phone: _______________",
            "Email: _______________",
            "Address: _____________",
            "Signature: ___________",
        ]
        font = _load_font(12)
        for line in fields:
            bbox = draw.textbbox((10, y_offset), line, font=font)
            draw.text((10, y_offset), line, fill="black", font=font)
            all_bboxes.append([10, y_offset, bbox[2], bbox[3]])
            all_texts.append(line)
            all_order.append(len(all_texts) / len(fields))
            all_line_ids.append(len(all_texts) - 1)
            full_text += line + "\n"
            y_offset += (bbox[3] - bbox[1]) + 8

    elif doc_type == "table":
        headers = "Item      Qty   Price"
        rows = [
            "Widget     5    $10.00",
            "Gadget     2    $25.50",
            "Tool       8     $5.99",
            "Part       3    $15.00",
        ]
        font_h = _load_font(12, bold=True)
        bbox = draw.textbbox((10, y_offset), headers, font=font_h)
        draw.text((10, y_offset), headers, fill="black", font=font_h)
        all_bboxes.append([10, y_offset, bbox[2], bbox[3]])
        all_texts.append(headers)
        all_order.append(0.0)
        all_line_ids.append(0)
        full_text += headers + "\n"
        y_offset += (bbox[3] - bbox[1]) + 6

        font_r = _load_font(12)
        for i, row in enumerate(rows):
            bbox = draw.textbbox((10, y_offset), row, font=font_r)
            draw.text((10, y_offset), row, fill="black", font=font_r)
            all_bboxes.append([10, y_offset, bbox[2], bbox[3]])
            all_texts.append(row)
            all_order.append((i + 1) / (len(rows) + 1))
            all_line_ids.append(i + 1)
            full_text += row + "\n"
            y_offset += (bbox[3] - bbox[1]) + 5

    return {
        "image": img,
        "text": full_text.strip(),
        "bounding_boxes": all_bboxes,
        "line_ids": all_line_ids,
        "reading_order": all_order,
        "difficulty": "medium",
        "doc_type": doc_type,
    }


# ── Dataset builder ──

class SyntheticOCRDataset:
    """Builds a complete synthetic OCR dataset with easy/medium/hard/document categories."""

    def __init__(self, output_dir: str = "data/synthetic_data"):
        self.output_dir = output_dir
        self.samples = []

    def _next_index(self, category: str) -> int:
        """Return the next available sample index for a category."""
        cat_dir = os.path.join(self.output_dir, category)
        existing = [f for f in os.listdir(cat_dir) if f.endswith(".png")] if os.path.isdir(cat_dir) else []
        indices = {int(f.replace("sample_", "").replace(".png", "")) for f in existing}
        i = 0
        while i in indices:
            i += 1
        return i

    def append_samples(
        self,
        n_easy: int = 300,
        n_medium: int = 300,
        n_hard: int = 300,
        n_document: int = 300,
    ):
        """Generate and append new samples without overwriting existing ones."""
        self.samples = []

        start = self._next_index("easy")
        print(f"Adding {n_easy} easy samples (index {start}–{start + n_easy - 1})...")
        for _ in range(n_easy):
            sample = generate_easy_sample() if random.random() < 0.5 else generate_easy_multi_line()
            sample["_index"] = start
            self.samples.append(sample)
            start += 1

        start = self._next_index("medium")
        print(f"Adding {n_medium} medium samples (index {start}–{start + n_medium - 1})...")
        for _ in range(n_medium):
            sample = generate_medium_sample()
            sample["_index"] = start
            self.samples.append(sample)
            start += 1

        start = self._next_index("hard")
        print(f"Adding {n_hard} hard samples (index {start}–{start + n_hard - 1})...")
        for _ in range(n_hard):
            sample = generate_hard_sample()
            sample["_index"] = start
            self.samples.append(sample)
            start += 1

        start = self._next_index("document")
        print(f"Adding {n_document} document samples (index {start}–{start + n_document - 1})...")
        for _ in range(n_document):
            sample = generate_document_sample()
            sample["difficulty"] = "document"
            sample["_index"] = start
            self.samples.append(sample)
            start += 1

        random.shuffle(self.samples)
        self._save_samples(append=True)

        stats = self.get_statistics()
        print(f"Append complete. Total: {stats['total_samples']} samples")
        print(f"  By difficulty: {stats['by_difficulty']}")

    def build_dataset(
        self,
        n_easy: int = 200,
        n_medium: int = 150,
        n_hard: int = 100,
        n_document: int = 100,
    ) -> list[dict]:
        """Build and save the synthetic dataset from scratch."""
        self.samples = []

        print(f"Generating {n_easy} easy samples...")
        for i in range(n_easy):
            sample = generate_easy_sample() if random.random() < 0.5 else generate_easy_multi_line()
            sample["_index"] = i
            self.samples.append(sample)

        print(f"Generating {n_medium} medium samples...")
        for i in range(n_medium):
            sample = generate_medium_sample()
            sample["_index"] = n_easy + i
            self.samples.append(sample)

        print(f"Generating {n_hard} hard samples...")
        for i in range(n_hard):
            sample = generate_hard_sample()
            sample["_index"] = n_easy + n_medium + i
            self.samples.append(sample)

        print(f"Generating {n_document} document samples...")
        for i in range(n_document):
            sample = generate_document_sample()
            sample["difficulty"] = "document"
            sample["_index"] = n_easy + n_medium + n_hard + i
            self.samples.append(sample)

        random.shuffle(self.samples)
        self._save_samples()
        return self.samples

    def _save_samples(self, append: bool = False):
        """Save generated images and metadata to disk.

        When append=True, uses sample['_index'] so existing files are not overwritten.
        """
        for sample in self.samples:
            category = sample["difficulty"]
            out_dir = os.path.join(self.output_dir, category)
            os.makedirs(out_dir, exist_ok=True)

            idx = sample.get("_index", None)
            name = f"sample_{idx:04d}" if idx is not None else f"sample_{len(os.listdir(out_dir)) if os.path.isdir(out_dir) else 0:04d}"
            img_path = os.path.join(out_dir, f"{name}.png")
            sample["image"].save(img_path)

            metadata = {
                "text": sample["text"],
                "bounding_boxes": sample["bounding_boxes"],
                "line_ids": sample["line_ids"],
                "reading_order": sample["reading_order"],
                "difficulty": sample["difficulty"],
            }
            if "doc_type" in sample:
                metadata["doc_type"] = sample["doc_type"]

            meta_path = os.path.join(out_dir, f"{name}.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

    def get_statistics(self) -> dict:
        difficulties = {}
        for s in self.samples:
            d = s["difficulty"]
            difficulties[d] = difficulties.get(d, 0) + 1
        return {"total_samples": len(self.samples), "by_difficulty": difficulties}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synthetic OCR dataset generator")
    parser.add_argument("--append", action="store_true", help="Add samples to existing dataset")
    parser.add_argument("--n-easy", type=int, default=300)
    parser.add_argument("--n-medium", type=int, default=300)
    parser.add_argument("--n-hard", type=int, default=300)
    parser.add_argument("--n-document", type=int, default=300)
    args = parser.parse_args()

    dataset = SyntheticOCRDataset(output_dir="data/synthetic_data")
    has_data = any(os.path.isdir(os.path.join(dataset.output_dir, d))
                   for d in ["easy", "medium", "hard", "document"])

    if has_data or args.append:
        dataset.append_samples(
            n_easy=args.n_easy, n_medium=args.n_medium,
            n_hard=args.n_hard, n_document=args.n_document,
        )
    else:
        samples = dataset.build_dataset(
            n_easy=args.n_easy, n_medium=args.n_medium,
            n_hard=args.n_hard, n_document=args.n_document,
        )
        print("Dataset generation complete!")
        print(dataset.get_statistics())
