import json
import regex as re
from collections.abc import Iterable, Iterator


PAT = (
    r"""'(?:[sdmt]|ll|ve|re)"""
    r"""| ?\p{L}+"""
    r"""| ?\p{N}+"""
    r"""| ?[^\s\p{L}\p{N}]+"""
    r"""|\s+(?!\S)"""
    r"""|\s+"""
)


class Tokenizer:
    """
    BPE tokenizer built on:
      - vocab: dict[int, bytes]
      - merges: list[tuple[bytes, bytes]] in *merge creation order*
    Supports:
      - encode(text) -> list[int]
      - encode_iterable(iterable[str]) -> Iterator[int]
      - decode(ids) -> str
      - optional special_tokens: list[str]
    """
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str]|None = None,
    ) -> None:
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges: list[tuple[bytes, bytes]] = list(merges)

        # Precompute: bytes -> id, pair -> rank (merge order)
        self.bytes_to_id: dict[bytes, int] = {
            token_bytes: tid for tid, token_bytes in self.vocab.items()
        }
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # Special tokens
        self.special_tokens_str_to_id: dict[str, int] = {}
        self.special_tokens_id_to_str: dict[int, str] = {}

        if special_tokens is not None:
            for tok in special_tokens:
                tok_bytes = tok.encode("utf-8")
                # If already in vocab, reuse its id
                if tok_bytes in self.bytes_to_id:
                    tok_id = self.bytes_to_id[tok_bytes]
                else:
                    # Append to vocab
                    tok_id = max(self.vocab.keys(), default=-1) + 1
                    self.vocab[tok_id] = tok_bytes
                    self.bytes_to_id[tok_bytes] = tok_id
                self.special_tokens_str_to_id[tok] = tok_id
                self.special_tokens_id_to_str[tok_id] = tok

    
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str]|None = None,
    ) -> "Tokenizer":
        """Construct a Tokenizer from serialized vocab and merges.
        """
        from tests.common import gpt2_bytes_to_unicode  # allowed in this assignment

        # Build byte decoder: unicode char -> byte value
        byte_encoder = gpt2_bytes_to_unicode()           # byte -> char
        byte_decoder = {ch: b for b, ch in byte_encoder.items()}  # char -> byte

        # --- Load vocab ---
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)  # typically: dict[str, int] (token -> id)

        vocab: dict[int, bytes] = {}
        for token_str, token_id in raw_vocab.items():
            # Convert GPT-2 token string back to bytes
            token_bytes = bytes([byte_decoder[c] for c in token_str])
            vocab[int(token_id)] = token_bytes

        # --- Load merges ---
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                a_str, b_str = line.split(" ")
                a_bytes = bytes([byte_decoder[c] for c in a_str])
                b_bytes = bytes([byte_decoder[c] for c in b_str])
                merges.append((a_bytes, b_bytes))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    

    def encode(self, text: str) -> list[int]:
        """Encode a string into token IDs.
        Respects special tokens: if a special token string appears in `text`,
        we emit its ID as a single token and we do *not* BPE-merge inside it.
        """
        if not text:
            return []

        # No special tokens: just BPE everything
        if not self.special_tokens_str_to_id:
            return self._encode_normal(text)

        ids: list[int] = []
        buffer: list[str] = []

        i = 0
        n = len(text)
        special_tokens = list(self.special_tokens_str_to_id.keys())

        while i < n:
            # Try to match any special token at position i
            matched_tok: str|None = None
            matched_id: int|None = None

            for tok in special_tokens:
                if text.startswith(tok, i):
                    # Choose longest match if overlapping special tokens exist
                    if matched_tok is None or len(tok) > len(matched_tok):
                        matched_tok = tok
                        matched_id = self.special_tokens_str_to_id[tok]

            if matched_tok is not None:
                # Flush buffered normal text before the special token
                if buffer:
                    normal_text = "".join(buffer)
                    ids.extend(self._encode_normal(normal_text))
                    buffer = []

                # Emit the special token as a single ID
                ids.append(matched_id)  # type: ignore[arg-type]
                i += len(matched_tok)
            else:
                # Normal character, accumulate
                buffer.append(text[i])
                i += 1

        # Flush trailing normal text
        if buffer:
            normal_text = "".join(buffer)
            ids.extend(self._encode_normal(normal_text))

        return ids
    

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encode an iterable of strings (e.g., lines from a file handle)
        and yield token IDs. This lets you tokenize very large files without
        loading the whole text into memory.

        Assumes each element of `iterable` is a valid Python `str` (so no
        splitting in the middle of a UTF-8 code unit).
        """
        for chunk in iterable:
            # Reuse `encode` so special tokens are handled consistently
            for tid in self.encode(chunk):
                yield tid


    def decode(self, ids: list[int]) -> str:
        """
        Decode token IDs back into text.

        If the bytes produced by concatenating token byte strings contain
        invalid UTF-8, we replace malformed bytes with U+FFFD by using
        errors='replace'.
        """
        if not ids:
            return ""

        out_bytes = bytearray()
        for tid in ids:
            try:
                token_bytes = self.vocab[tid]
            except KeyError:
                # If a random unknown id is supplied, treat as replacement char
                print("Unknown token id %s during decode; replacing.", tid)
                # U+FFFD in UTF-8
                out_bytes.extend("\uFFFD".encode("utf-8"))
                continue
            out_bytes.extend(token_bytes)

        return out_bytes.decode("utf-8", errors="replace")


    def _pretokenize(self, text: str) -> list[bytes]:
        """
        Pre-tokenize using the same PAT as in training.
        Returns list of UTF-8 byte sequences (one per pre-token).
        """
        if not text:
            return []
        tokens: list[bytes] = []
        for m in re.finditer(PAT, text):
            s = m.group(0)
            tokens.append(s.encode("utf-8"))
        return tokens


    def _encode_normal(self, text: str) -> list[int]:
        """
        Encode text with *no* special token handling (used internally).
        """
        ids: list[int] = []
        for token_bytes in self._pretokenize(text):
            ids.extend(self._bpe_encode_pretoken(token_bytes))
        return ids


    def _bpe_encode_pretoken(self, token_bytes: bytes) -> list[int]:
        """
        Apply BPE merges to a single pre-token represented as bytes.
        We:
          1. Initialize the sequence as single-byte tokens.
          2. Repeatedly merge the pair with the lowest merge rank
             until no pair in the sequence appears in `merge_ranks`.
          3. Map the resulting byte chunks to token IDs via `bytes_to_id`.
        """
        if not token_bytes:
            return []

        # Start from single-byte tokens
        seq: list[bytes] = [bytes([b]) for b in token_bytes]

        if len(seq) == 1:
            # Single byte, must already exist as a vocab entry
            return [self.bytes_to_id[seq[0]]]

        while True:
            # Find best (lowest rank) pair in current sequence
            best_pair = None
            best_rank = None

            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                # No more applicable merges
                break

            # Merge all occurrences of best_pair in the sequence
            new_seq: list[bytes] = []
            i = 0
            while i < len(seq):
                if (
                    i < len(seq) - 1
                    and seq[i] == best_pair[0]
                    and seq[i + 1] == best_pair[1]
                ):
                    new_seq.append(seq[i] + seq[i + 1])
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            seq = new_seq

            if len(seq) == 1:
                break

        # Map final byte chunks to IDs
        ids: list[int] = []
        for chunk in seq:
            tid = self.bytes_to_id.get(chunk)
            if tid is None:
                # This shouldn't happen if vocab/merges are consistent,
                # but be defensive.
                raise KeyError(f"Chunk {chunk!r} not found in vocab.")
            ids.append(tid)
        return ids