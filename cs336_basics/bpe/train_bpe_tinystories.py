import os
import json
import time
from .train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode


TINYSTORIES_SPECIAL = "<|endoftext|>"
VOCAB_SIZE = 10000


def _bytes_to_gpt2_token(bseq: bytes, byte_encoder: dict[int, str]) -> str:
    """Convert a bytes sequence into GPT-2-style token string."""
    return "".join(byte_encoder[b] for b in bseq)


def train_bpe_tinystories(
    input_path: str,
    output_dir: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer on TinyStories and serialize vocab/merges.
    """
    os.makedirs(output_dir, exist_ok=True)

    byte_encoder = gpt2_bytes_to_unicode()

    special_tokens = [TINYSTORIES_SPECIAL]
    desired_num_chunks = os.cpu_count() or 1

    start = time.perf_counter()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=VOCAB_SIZE,
        special_tokens=special_tokens,
        desired_num_chunks=desired_num_chunks
    )
    end = time.perf_counter()
    elapsed_sec = end - start

    merges_path = os.path.join(output_dir, "merges.txt")
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            ta = _bytes_to_gpt2_token(a, byte_encoder)
            tb = _bytes_to_gpt2_token(b, byte_encoder)
            f.write(f"{ta} {tb}\n")

    vocab_path = os.path.join(output_dir, "vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        vocab_json = {
            _bytes_to_gpt2_token(token_bytes, byte_encoder): token_id
            for token_id, token_bytes in vocab.items()
        }
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    print(f"Trained TinyStories BPE in {elapsed_sec/60:.2f} minutes.")
    print(f"Vocab size: {len(vocab)} (target {VOCAB_SIZE})")
    print(f"Number of merges: {len(merges)}")

    return vocab, merges

if __name__ == "__main__":
    input_path = "data/TinyStoriesV2-GPT4-train.txt"
    output_dir = "data/tokenizer/tinystories"
    vocab, merges = train_bpe_tinystories(
        input_path=input_path,
        output_dir=output_dir,
    )