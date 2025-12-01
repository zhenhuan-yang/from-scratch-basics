import os
import json
import time
import resource
from .train_bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode


TINYSTORIES_SPECIAL = "<|endoftext|>"
VOCAB_SIZE = 32000


def _bytes_to_gpt2_token(bseq: bytes, byte_encoder: dict[int, str]) -> str:
    """Convert a bytes sequence into GPT-2-style token string."""
    return "".join(byte_encoder[b] for b in bseq)


def train_bpe_owt(
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

    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_kb = usage.ru_maxrss

    longest_token_bytes = max(vocab.values(), key=len)
    longest_token_len = len(longest_token_bytes)  # in bytes
    longest_token_str = _bytes_to_gpt2_token(longest_token_bytes, byte_encoder)

    print(f"Training time: {elapsed_sec / 60:.2f} minutes.")
    print(f"Peak memory (approx): {peak_kb / 1024.0 / 1024.0:.2f} GB.")
    print(f"Longest token length: {longest_token_len} bytes.")
    print(f"Longest token (GPT-2 printable): {repr(longest_token_str)}")

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

    return vocab, merges

if __name__ == "__main__":
    input_path = "data/owt_train.txt"
    output_dir = "data/tokenizer/owt"
    vocab, merges = train_bpe_owt(
        input_path=input_path,
        output_dir=output_dir,
    )