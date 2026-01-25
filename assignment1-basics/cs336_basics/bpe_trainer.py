from __future__ import annotations

import os
from dataclasses import dataclass
from typing import BinaryIO, Iterable, Iterator
from collections import Counter, defaultdict
import json
import regex as re
import multiprocessing as mp
from tqdm import tqdm

GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    将文件切分成若干块(chunk)，每块可以独立做计数/处理
    会尝试把切分边界对齐到指定的特殊分隔符(split_special_token)上
    如果多个边界最终重合，可能返回的 chunk 数量会少于 desired_num_chunks

    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    # chunk 采用左闭右开区间: [previous_idx, last_idx)
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Splits the input text into segments based on the provided special tokens."""
    if not special_tokens:
        return [text]

    # 这里不需要对特殊标记进行按长度排序，因为我们不保留它们
    pattern = "(" + "|".join(map(re.escape, special_tokens)) + ")"
    parts = re.split(pattern, text)
    return [p for p in parts if p and p not in special_tokens] # 不保留特殊标记

def _pretokenize(text: str) -> list[str]:
    """Pre-tokenizes the input text into byte-level tokens using the GPT-2 regex pattern."""
    return [m.group(0) for m in re.finditer(GPT2_PATTERN, text)]

Pair = tuple[bytes, bytes]
Word = tuple[bytes, ...]

def _pairs_in_word(word: Word) -> Counter[Pair]:
    """计算一个 Word 中相邻 byte 对出现的次数"""
    if len(word) < 2:
        return Counter()
    return Counter(zip(word, word[1:]))

@dataclass(frozen=True)
class _ChunkJob:
    path: str
    start: int
    end: int
    special_tokens: list[str]

def _count_chunk(job: _ChunkJob) -> Counter[tuple[bytes, ...]]:
    """Single process function for pretokenization and counting byte-level tokens in a chunk."""
    with open(job.path, "rb") as f:
        f.seek(job.start, os.SEEK_SET)
        text = f.read(job.end - job.start).decode("utf-8", errors="ignore")

    word_freq = Counter()
    for seg in _split_on_special_tokens(text, job.special_tokens):
        for pretok in _pretokenize(seg):
            word = tuple(bytes([b]) for b in pretok.encode("utf-8"))
            word_freq[word] += 1

    return word_freq

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    """Trains a Byte Pair Encoding (BPE) tokenizer on the given input text file."""

    assert vocab_size >= 256 + len(set(special_tokens)), "Vocabulary size must be at least 256 plus the number of special tokens."

    # Initialize the base vocabulary with single-byte tokens (0-255)
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # Add special tokens to the vocabulary
    existing_tokens = set(vocab.values())
    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode('utf-8')
            if token_bytes not in existing_tokens:
                vocab[len(vocab)] = token_bytes
                existing_tokens.add(token_bytes)

    # Pretokenization phase
    with open(input_path, 'rb') as f:
        num_processes = 8
        boundaries = _find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    jobs = [
        _ChunkJob(input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    # 并行化处理Pretokenization阶段
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_count_chunk, jobs, chunksize=1), total=len(jobs), desc="Pretokenize"))

    # 合并所有chunk的计数结果
    total = Counter()
    for c in results:
        total.update(c)
    word_freqs: dict[tuple[bytes, ...], int] = defaultdict(int, total)

    # BPE merging process
    bpe_merges: list[tuple[bytes, bytes]] = []

    word_pair_counts: dict[Word, Counter[Pair]] = {}          # Word -> Counter of pairs in that word
    pair_counts: Counter[Pair] = Counter()                    # Pair -> total count across all words
    pair2words: dict[Pair, set[Word]] = defaultdict(set)      # Pair -> set of words containing that pair

    # Initialize pair counts from the initial words
    for word, freq in word_freqs.items():
        pairs = _pairs_in_word(word)
        word_pair_counts[word] = pairs
        for pair, count in pairs.items():
            pair_counts[pair] += count * freq
            pair2words[pair].add(word)

    # BPE merge iterations
    initial_vocab_len = len(vocab)
    target_merges = vocab_size - initial_vocab_len
    pbar = tqdm(total=target_merges, desc="BPE merges")

    while len(vocab) < vocab_size:
        # Find the most frequent pair
        best_count = max(pair_counts.values())
        best_pairs = [pair for pair, count in pair_counts.items() if count == best_count]
        best_pair = max(best_pairs) # Tie-breaking by lexicographical order
        
        if best_count == 0:
            break  # No more pairs to merge

        # Add the best pair to the vocabulary
        a, b = best_pair
        new_token = a + b

        if new_token in existing_tokens:
            break

        vocab[len(vocab)] = new_token
        existing_tokens.add(new_token)
        bpe_merges.append(best_pair)

        # Get the list of words affected by this merge
        affected_words = list(pair2words.get(best_pair, set()))
        if not affected_words:
            pair_counts.pop(best_pair, None)
            pbar.update(1)
            continue

        new_words_added: dict[Word, int] = defaultdict(int)

        # Update affected words
        for word in affected_words:
            # Get the frequency of this word
            word_freq = word_freqs.get(word, 0)
            if word_freq == 0:
                continue
            
            # Remove old pair counts for this word
            old_local = word_pair_counts[word]
            for pair, count in old_local.items():
                pair_counts[pair] -= count * word_freq
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                    pair2words.pop(pair, None)
                else:
                    pair2words[pair].discard(word)

            # Create the new word by merging the best pair
            new_word: List[bytes] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)

            new_words_added[new_word] += word_freq

            del word_freqs[word]
            del word_pair_counts[word]

        # Add new words and update pair counts
        for new_word, freq in new_words_added.items():
            word_freqs[new_word] += freq
            pairs = _pairs_in_word(new_word)
            word_pair_counts[new_word] = pairs

            for pair, count in pairs.items():
                pair_counts[pair] += count * freq
                pair2words[pair].add(new_word)

        if best_pair in pair2words and not pair2words[best_pair]:
            pair2words.pop(best_pair, None)
        if best_pair in pair_counts and pair_counts[best_pair] <= 0:
            pair_counts.pop(best_pair, None)

        pbar.update(1)
    
    pbar.close()

    return vocab, bpe_merges

def save_bpe_model(vocab: dict[int, bytes], bpe_merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str) -> None:
    # vocab: int -> bytes  =>  str(int) -> list[int]
    vocab_json = {str(i): list(b) for i, b in vocab.items()}

    # merges: (bytes, bytes) => [[list[int], list[int]], ...]
    merges_json = [[list(a), list(b)] for a, b in bpe_merges]

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        json.dump(merges_json, f, ensure_ascii=False)

def load_bpe_model(vocab_path: str, merges_path: str) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_json = json.load(f)
    vocab = {int(i): bytes(b) for i, b in vocab_json.items()}

    with open(merges_path, "r", encoding="utf-8") as f:
        merges_json = json.load(f)
    bpe_merges = [(bytes(a), bytes(b)) for a, b in merges_json]

    return vocab, bpe_merges

def train_bpe_tinystories():
    input_path = "./data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    vocab, bpe_merges = train_bpe(input_path, vocab_size, special_tokens)
    save_bpe_model(vocab, bpe_merges, "vocab.json", "merges.json")

def train_bpe_expts_owt():
    input_path = "./data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    vocab, bpe_merges = train_bpe(input_path, vocab_size, special_tokens)
    save_bpe_model(vocab, bpe_merges, "vocab_owt.json", "merges_owt.json")

if __name__ == "__main__":
    # train_bpe_tinystories()
    train_bpe_expts_owt()