"""OCR-specific character-first BPE / byte-aware tokenizer.

Supports:
    - uppercase/lowercase characters
    - digits
    - punctuation
    - whitespace representation
    - common OCR symbols
    - frequent character sequences (BPE merges)
    - special tokens (PAD, BOS, EOS, UNK)

The tokenizer is character-first: individual characters form the base vocabulary,
while frequent character pairs/n-grams are learned as larger BPE tokens to
compress sequences and handle rare/unusual characters robustly.
"""

import json
import os
from collections import Counter, defaultdict
from typing import List


# ── Base character vocabulary ──
BASE_CHARS = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?()[]{}/\\\"'-–—@#$%&*+=<>~`|°_€£¥"
    "\n\t\r"
)

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<BOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3,
}


class OCRTokenizer:
    """Character-first BPE tokenizer for OCR text.

    The vocabulary is built in two stages:
        1. All base characters are added as individual tokens
        2. Frequent character pairs are merged into BPE tokens iteratively

    Special tokens occupy indices 0–3. The remaining indices are for
    characters and BPE merges.
    """

    def __init__(self, vocab_size: int = 2000, max_bpe_merges: int = None):
        self.vocab_size = vocab_size
        self.max_bpe_merges = max_bpe_merges or (vocab_size - len(SPECIAL_TOKENS))

        # Build vocabulary: special tokens + base characters + BPE merges
        self.token_to_id = dict(SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}

        # Add base characters
        for ch in BASE_CHARS:
            token = f"<{ch}>" if ch in "\n\t\r " else ch
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        # BPE merges learned during training
        self.bpe_merges = {}  # (token_a, token_b) → merged_token
        self.bpe_ranks = {}   # merged_token → rank

        self._fitted = False

    # ── Training ──

    def train(self, texts: List[str], num_merges: int = None):
        """Train BPE merges on a corpus of text.

        Args:
            texts: List of strings to train on
            num_merges: Number of BPE merge operations to perform
        """
        num_merges = num_merges or self.max_bpe_merges

        # Start with character-level tokenization
        word_freqs = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                word_freqs[word] += 1

        # Initialize vocab with characters from all words
        vocab = set(self.token_to_id.keys())
        word_chars = {}
        for word, freq in word_freqs.items():
            chars = list(word)
            word_chars[word] = chars
            for ch in chars:
                token = self._char_to_token(ch)
                vocab.add(token)

        # Learn BPE merges
        self.bpe_merges = {}
        self.bpe_ranks = {}
        merge_count = 0

        while merge_count < num_merges:
            # Count pairs
            pair_freqs = Counter()
            for word, chars in word_chars.items():
                for i in range(len(chars) - 1):
                    pair = (chars[i], chars[i + 1])
                    pair_freqs[pair] += word_freqs.get(word, 0)

            if not pair_freqs:
                break

            # Find most frequent pair
            best_pair, best_freq = pair_freqs.most_common(1)[0]
            if best_freq < 2:
                break  # No more meaningful merges

            # Create merged token
            merged = best_pair[0] + best_pair[1]
            if merged in vocab:
                # Already exists, skip
                break

            new_idx = len(self.token_to_id)
            self.token_to_id[merged] = new_idx
            self.id_to_token[new_idx] = merged
            vocab.add(merged)
            self.bpe_merges[best_pair] = merged
            self.bpe_ranks[merged] = merge_count
            merge_count += 1

            # Update word_chars
            new_word_chars = {}
            for word, chars in word_chars.items():
                new_chars = []
                i = 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == best_pair:
                        new_chars.append(merged)
                        i += 2
                    else:
                        new_chars.append(chars[i])
                        i += 1
                new_word_chars[word] = new_chars
            word_chars = new_word_chars

        self._fitted = True
        self.vocab_size = len(self.token_to_id)

    def _char_to_token(self, ch: str) -> str:
        """Convert a character to its token representation."""
        if ch == " ":
            return "< >"
        if ch == "\n":
            return "<\\n>"
        if ch == "\t":
            return "<\\t>"
        if ch == "\r":
            return "<\\r>"
        return ch

    # ── Encoding ──

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        """Encode a string into token IDs.

        Args:
            text: Input string
            add_bos: Prepend BOS token
            add_eos: Append EOS token
        Returns:
            List of token IDs
        """
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)

        return ids

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into BPE tokens (greedy longest-match)."""
        if not self._fitted:
            # Fall back to character-level
            return [self._char_to_token(ch) for ch in text]

        words = text.split()
        all_tokens = []
        for word in words:
            chars = list(word)
            tokens = self._merge_bpe(chars)
            all_tokens.extend(tokens)
            all_tokens.append(" ")  # word separator

        if all_tokens and all_tokens[-1] == " ":
            all_tokens = all_tokens[:-1]  # Remove trailing space

        return all_tokens

    def _merge_bpe(self, chars: List[str]) -> List[str]:
        """Apply greedy BPE merging to a list of characters."""
        if not self.bpe_merges:
            return chars

        tokens = list(chars)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(tokens) - 1:
                pair = (tokens[i], tokens[i + 1])
                if pair in self.bpe_merges:
                    tokens[i] = self.bpe_merges[pair]
                    tokens.pop(i + 1)
                    changed = True
                else:
                    i += 1
        return tokens

    def _find_best_pair(self, tokens: List[str]) -> tuple | None:
        """Find the highest-ranked BPE merge pair in a token sequence."""
        best_pair = None
        best_rank = float('inf')
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in self.bpe_ranks and self.bpe_ranks[pair] < best_rank:
                best_rank = self.bpe_ranks[pair]
                best_pair = pair
        return best_pair

    # ── Decoding ──

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs back to text.

        Args:
            ids: List of token IDs
        Returns:
            Decoded string
        """
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                tokens.append("<UNK>")

        text = ""
        for token in tokens:
            if token == "< >":
                text += " "
            elif token == "<\\n>":
                text += "\n"
            elif token == "<\\t>":
                text += "\t"
            elif token == "<\\r>":
                text += "\r"
            elif token.startswith("<") and token.endswith(">") and len(token) > 1:
                # Special token not recognized as space/newline
                pass  # Skip or handle specially
            elif token in ("<BOS>", "<EOS>", "<PAD>", "<UNK>"):
                pass  # Skip special tokens in output
            else:
                text += token

        return text

    # ── Properties ──

    @property
    def pad_token_id(self) -> int:
        return SPECIAL_TOKENS["<PAD>"]

    @property
    def bos_token_id(self) -> int:
        return SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return SPECIAL_TOKENS["<EOS>"]

    @property
    def unk_token_id(self) -> int:
        return SPECIAL_TOKENS["<UNK>"]

    def save(self, path: str):
        """Save tokenizer to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "bpe_ranks": self.bpe_ranks,
            "vocab_size": self.vocab_size,
            "max_bpe_merges": self.max_bpe_merges,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OCRTokenizer":
        """Load tokenizer from disk."""
        with open(path) as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"], max_bpe_merges=data["max_bpe_merges"])
        tokenizer.token_to_id = {k: v for k, v in data["token_to_id"].items()}
        tokenizer.id_to_token = {v: k for k, v in data["id_to_token"].items()}
        tokenizer.bpe_ranks = data["bpe_ranks"]
        tokenizer._fitted = True
        return tokenizer

    def __len__(self) -> int:
        return len(self.token_to_id)

    def __repr__(self):
        return f"OCRTokenizer(vocab_size={len(self)}, bpe_merges={len(self.bpe_merges)})"
