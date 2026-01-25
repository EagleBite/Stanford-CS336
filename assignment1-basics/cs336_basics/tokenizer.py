from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, Iterator
import regex as re

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Splits the input text into segments based on the provided special tokens."""
    if not special_tokens:
        return [text]

    special_tokens = sorted(special_tokens, key=len, reverse=True) # 按长度排序以优先匹配较长的标记
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p] # 保留特殊标记

def _pretokenize(text: str) -> list[str]:
    """Pre-tokenizes the input text into byte-level tokens using the GPT-2 regex pattern."""
    return [m.group(0) for m in re.finditer(GPT2_PATTERN, text)]

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens if special_tokens else []
        
        # Add special tokens to the vocabulary if they are not already present
        existing_tokens = set(vocab.values())
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in existing_tokens:
                self.vocab[len(self.vocab)] = token_bytes
                existing_tokens.add(token_bytes)

        # Create a mapping from bytes to IDs
        self.bytes_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab = {int(i): bytes(v) for i, v in vocab_json.items()}

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges_json = json.load(f)
        merges = [(bytes(a), bytes(b)) for a, b in merges_json]

        return cls(vocab, merges, special_tokens)

    def _merge_in_order(self, symbols: list[bytes]) -> list[bytes]:
        """Merges byte pairs in the input list according to the BPE merges."""
        symbols = symbols.copy()

        while True:
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            candidate_pairs = [(pair, self.merges[pair]) for pair in pairs if pair in self.merges]

            if not candidate_pairs:
                break

            # Find the earliest merge pair
            best_pair, _ = min(candidate_pairs, key=lambda x: x[1])

            # Merge the best pair
            new_symbols = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == best_pair:
                    new_symbols.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        return symbols

    def encode(self, text: str) -> list[int]:
        ids = []

        for seg in _split_on_special_tokens(text, self.special_tokens):
            if seg in self.special_tokens:
                b = seg.encode("utf-8")
                ids.append(self.bytes_to_id[b])
                continue
            for pretok in _pretokenize(seg):
                symbols = tuple(bytes([b]) for b in pretok.encode("utf-8"))
                merged_symbols = self._merge_in_order(list(symbols))
                for symbol in merged_symbols:
                    ids.append(self.bytes_to_id[symbol])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        bytes_list = [self.vocab[id] for id in ids]
        return b"".join(bytes_list).decode("utf-8", errors="replace")

