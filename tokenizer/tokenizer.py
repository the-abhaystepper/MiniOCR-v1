"""Character-first BPE tokenizer for OCR text."""

import json
import os
from collections import Counter
from typing import List

# Special token IDs are fixed; all remaining indices are characters/BPE merges.
SPECIAL_TOKENS = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3}

BASE_CHARS = (
    " abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,;:!?()[]{}/\\\"'-–—@#$%&*+=<>~`|°_€£¥"
    "\n\t\r"
)


class OCRTokenizer:
    """Character-first BPE tokenizer.

    Base vocabulary consists of ASCII/extended characters.
    Frequent character pairs are iteratively merged into BPE tokens.
    """

    def __init__(self, vocab_size: int = 2000, max_bpe_merges: int = None):
        self.vocab_size = vocab_size
        self.max_bpe_merges = max_bpe_merges or (vocab_size - len(SPECIAL_TOKENS))
        self.token_to_id = dict(SPECIAL_TOKENS)
        self.id_to_token = {v: k for k, v in SPECIAL_TOKENS.items()}

        for ch in BASE_CHARS:
            token = f"<{ch}>" if ch in "\n\t\r " else ch
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token

        self.bpe_merges: dict[tuple, str] = {}
        self.bpe_ranks: dict[str, int] = {}
        self._fitted = False

    def train(self, texts: List[str], num_merges: int = None):
        num_merges = num_merges or self.max_bpe_merges
        word_freqs = Counter()
        for text in texts:
            for word in text.split():
                word_freqs[word] += 1

        vocab = set(self.token_to_id.keys())
        word_chars = {}
        for word, freq in word_freqs.items():
            chars = list(word)
            word_chars[word] = chars
            for ch in chars:
                vocab.add(self._char_to_token(ch))

        self.bpe_merges = {}
        self.bpe_ranks = {}
        merge_count = 0

        while merge_count < num_merges:
            pair_freqs = Counter()
            for word, chars in word_chars.items():
                for i in range(len(chars) - 1):
                    pair_freqs[(chars[i], chars[i + 1])] += word_freqs.get(word, 0)

            if not pair_freqs:
                break
            (best_a, best_b), best_freq = pair_freqs.most_common(1)[0]
            if best_freq < 2:
                break

            merged = best_a + best_b
            if merged in vocab:
                break

            new_idx = len(self.token_to_id)
            self.token_to_id[merged] = new_idx
            self.id_to_token[new_idx] = merged
            vocab.add(merged)
            self.bpe_merges[(best_a, best_b)] = merged
            self.bpe_ranks[merged] = merge_count
            merge_count += 1

            # Re-tokenize words with the new merge
            new_word_chars = {}
            for word, chars in word_chars.items():
                new_chars, i = [], 0
                while i < len(chars):
                    if i < len(chars) - 1 and (chars[i], chars[i + 1]) == (best_a, best_b):
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
        return {" ": "< >", "\n": "<\\n>", "\t": "<\\t>", "\r": "<\\r>"}[ch] if ch in "\n\t\r " else ch

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = self._tokenize(text)
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def _tokenize(self, text: str) -> List[str]:
        if not self._fitted:
            return [self._char_to_token(ch) for ch in text]
        all_tokens = []
        for word in text.split():
            all_tokens.extend(self._merge_bpe(list(word)))
            all_tokens.append(" ")
        if all_tokens and all_tokens[-1] == " ":
            all_tokens.pop()
        return all_tokens

    def _merge_bpe(self, chars: List[str]) -> List[str]:
        if not self.bpe_merges:
            return chars
        tokens = list(chars)
        changed = True
        while changed:
            changed = False
            i = 0
            while i < len(tokens) - 1:
                if (tokens[i], tokens[i + 1]) in self.bpe_merges:
                    merged = self.bpe_merges[(tokens[i], tokens[i + 1])]
                    tokens[i] = merged
                    tokens.pop(i + 1)
                    changed = True
                else:
                    i += 1
        return tokens

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for idx in ids:
            if idx in self.id_to_token:
                tokens.append(self.id_to_token[idx])
            else:
                tokens.append("<UNK>")

        parts = []
        for t in tokens:
            if t == "< >":
                parts.append(" ")
            elif t == "<\\n>":
                parts.append("\n")
            elif t == "<\\t>":
                parts.append("\t")
            elif t == "<\\r>":
                parts.append("\r")
            elif t in SPECIAL_TOKENS or (t.startswith("<") and t.endswith(">")):
                continue
            else:
                parts.append(t)
        return "".join(parts)

    @property
    def pad_token_id(self): return SPECIAL_TOKENS["<PAD>"]
    @property
    def bos_token_id(self): return SPECIAL_TOKENS["<BOS>"]
    @property
    def eos_token_id(self): return SPECIAL_TOKENS["<EOS>"]
    @property
    def unk_token_id(self): return SPECIAL_TOKENS["<UNK>"]

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({
                "token_to_id": self.token_to_id,
                "id_to_token": self.id_to_token,
                "bpe_ranks": self.bpe_ranks,
                "vocab_size": self.vocab_size,
                "max_bpe_merges": self.max_bpe_merges,
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "OCRTokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"], max_bpe_merges=data["max_bpe_merges"])
        tok.token_to_id = data["token_to_id"]
        tok.id_to_token = data["id_to_token"]
        tok.bpe_ranks = data["bpe_ranks"]
        tok._fitted = True
        return tok

    def __len__(self): return len(self.token_to_id)
    def __repr__(self): return f"OCRTokenizer(vocab_size={len(self)}, bpe_merges={len(self.bpe_merges)})"
