import os
import time
import logging
from typing import BinaryIO
from collections import defaultdict
import regex as re # faster than built-in `re`
from multiprocessing import Pool


logger = logging.getLogger(__name__)

# https://github.com/openai/tiktoken/pull/234/files
PAT = (
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)
# Prebuild 256 single-byte tokens to reuse (avoids allocating bytes([b]) repeatedly)
SINGLE_BYTE_TOKENS: list[bytes] = [bytes([i]) for i in range(256)]

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
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


def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """Split text on special tokens, dropping the special tokens."""
    if not special_tokens:
        return [text]
    
    # Build a regex like <\|endoftext\|>
    # Each token is escaped so that it is treated as a literal string to prevent regex metacharacter interpretation
    escaped = [re.escape(tok) for tok in special_tokens]
    pattern = "|".join(escaped)
    parts = re.split(pattern, text)

    return [p for p in parts if p.strip() != ""]


def pretokenize_and_count(text: str) -> dict[tuple[bytes], int]:
    """Pretokenize text and count the frequency.
    Each pretoken is cast as a sequence of bytes and mapped to the pre-token frequency.
    """
    freq: dict[tuple[bytes, ...], int] = defaultdict(int)
    
    # use re.finditer to avoid storing the pre-tokenized words as you construct your mapping from pre-tokens to their counts
    for match in re.finditer(PAT, text):
        pretoken_str = match.group(0)
        if not pretoken_str:
            continue

        pretoken_bytes = pretoken_str.encode("utf-8")
        if not pretoken_bytes:
            continue

        pretoken_repr = tuple(SINGLE_BYTE_TOKENS[b] for b in pretoken_bytes)
        freq[pretoken_repr] += 1

    return freq


def _build_word_freq_for_chunk(
    input_path: str | os.PathLike,
    chunk_start: int,
    chunk_end: int,
    special_tokens: list[str],
) -> dict[tuple[bytes], int]:
    """Worker function to run in a sepearte process (or serially).
    It reads chunk bytes in the range [start, end).
    Then it split_on_special_tokens and pretokenize_and_count on each segment and accumulates
    """
    freq: dict[tuple[bytes, ...], int] = defaultdict(int)

    # Each worker re-open the file for safety
    with open(input_path, "rb") as f:
        f.seek(chunk_start)
        chunk_bytes = f.read(chunk_end - chunk_start)

    chunk_str = chunk_bytes.decode("utf-8", errors="ignore")
    segment_str_list = split_on_special_tokens(chunk_str, special_tokens)

    for segment_str in segment_str_list:
        seg_freq = pretokenize_and_count(segment_str)
        for repr, count in seg_freq.items():
            freq[repr] += count
    
    return freq


def combine_chunk_freq(freq_list: list[dict[tuple[bytes, ...], int]]) -> dict[tuple[bytes, ...], int]:
    """Combine a list of freq dicts into one by summing counts.
    """
    combined_freq:  dict[tuple[bytes, ...], int] = defaultdict(int)

    for freq in freq_list:
        for repr, count in freq.items():
            combined_freq[repr] += count
    
    return combined_freq


def pretokenize_file_in_parallel(
    input_path: str | os.PathLike,
    desired_num_chunks: int,
    special_tokens: list[str],
) -> dict[tuple[bytes, ...], int]:
    """Pretokenize a file in parallel and return pre-token frequency
    """
    input_path = os.fspath(input_path)
    if special_tokens:
        special_token_bytes = special_tokens[0].encode("utf-8")
    else:
        special_token_bytes = None

    with open(input_path, "rb") as f:
        if special_token_bytes is not None:
            boundaries = find_chunk_boundaries(f, desired_num_chunks, special_token_bytes)
        else:
            # No special token, Get total file size in bytes and assign single chunk
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            boundaries = [0, file_size]

    chunks: list[tuple[int, int]] = list(zip(boundaries[:-1], boundaries[1:]))

    # Limit workers to number of ranges (otherwise some workers idle).
    num_workers = min(desired_num_chunks, len(chunks))

    worker_args = [(input_path, chunk_start, chunk_end, special_tokens) for (chunk_start, chunk_end) in chunks]
    with Pool(processes=num_workers) as pool:
        partial_freqs: list[dict[tuple[bytes, ...], int]] = pool.starmap(_build_word_freq_for_chunk, worker_args)

    return combine_chunk_freq(partial_freqs)