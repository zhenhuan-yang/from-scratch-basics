import os
import time
import logging
from collections import defaultdict
from .pretokenization import pretokenize_file


logger = logging.getLogger(__name__)


def build_initial_vocab() -> dict[int, bytes]:
    """Create the initial vocab with the 256 single-byte tokens.
    """
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    
    return vocab


def compute_pair_counts(
    freq: dict[tuple[bytes, ...], int]
) -> dict[tuple[bytes, bytes], int]:
    """Compute frequency weighted counts of adjacent pairs over the whole corpus.
    Also keeps a set of sequences (i.e. tuple[bytes, ...]) that contain that pair
    """
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

    # Since corpus is pretokenized, we just sum the pair freq count from the pre-token
    # Otherwise we need to compute every pair throughout the corpus
    for repr, count in freq.items():
        if len(repr) < 2:
            continue
        
        for i in range(len(repr) - 1):
            pair = (repr[i], repr[i + 1])
            pair_counts[pair] += count

    return pair_counts


def build_pair_stats(
    freq: dict[tuple[bytes, ...], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]:
    """
    Build:
      - pair_counts[pair] = total frequency in corpus
      - pair_to_seqs[pair] = set of seqs (tuples) that contain that pair
    """
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)

    for seq, count in freq.items():
        if len(seq) < 2:
            continue
        # Walk adjacent pairs
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i + 1])
            pair_counts[pair] += count
            pair_to_seqs[pair].add(seq)

    return pair_counts, pair_to_seqs


def find_most_freq_pair(pair_counts: dict[tuple[bytes, bytes], int]) -> tuple[bytes, bytes]:
    """Pick the most freq pair with tie breaking by lexicographically greater pair.
    """
    pair, count = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0])) # (count, lexicographic)
    
    return pair, count


def merge_pair_in_sequence(
    seq: tuple[bytes, ...],
    pair: tuple[bytes, bytes],
    new_token: bytes
) -> tuple[bytes, ...]:
    """Merge all occurrences of `pair` into `new_token` in a single sequence.
    """
    a, b = pair
    out: list[bytes] = []
    i = 0
    n = len(seq)

    while i < n:
        if i < n - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new_token)
            i += 2
        else:
            out.append(seq[i])
            i += 1

    return tuple(out)


def apply_merge(
    freq: dict[tuple[bytes, ...], int],
    most_freq_pair: tuple[bytes, bytes]
) -> bytes:
    """Apply merge of `most_freq_pair` to all sequences that contains it,
    and update pair_counts / pair_to_seqs *incrementally*.

    Corpus is always dict[tuple[bytes, ...], int]; we mutate it in-place.

    Returns:
        new_token (bytes): the merged token a + b
    """
    a, b = most_freq_pair
    new_token = a + b

    # We'll build a new corpus dict
    new_freq: dict[tuple[bytes, ...], int] = defaultdict(int)

    for old_repr, old_count in freq.items():
        new_repr = merge_pair_in_sequence(old_repr, most_freq_pair, new_token)
        new_freq[new_repr] = new_freq.get(new_repr, 0) + old_count

    # Replace corpus dict in-place
    freq.clear()
    freq.update(new_freq)

    return new_token


def apply_merge_incremental(
    freq: dict[tuple[bytes, ...], int],
    most_freq_pair: tuple[bytes, bytes],
    pair_counts: dict[tuple[bytes, bytes], int],
    pair_to_seqs: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]],
) -> bytes:
    """
    Apply merge of `most_freq_pair` to all sequences that contain it,
    updating:
      - freq (corpus)
      - pair_counts
      - pair_to_seqs

    This is mathematically equivalent to recomputing from scratch, just faster.
    """
    a, b = most_freq_pair
    new_token = a + b

    # Get all sequences that contain this pair (copy because we'll mutate structures)
    affected_seqs = list(pair_to_seqs.pop(most_freq_pair, []))
    if not affected_seqs:
        # Shouldn't really happen if pair_counts was > 0, but be robust
        return new_token

    # 1) Remove contribution of OLD sequences from pair_counts / pair_to_seqs
    #    and compute their merged versions.
    merged_seqs: dict[tuple[bytes, ...], int] = defaultdict(int)

    for old_seq in affected_seqs:
        # This seq might already have been merged in a prior step (collapsed to another seq)
        old_count = freq.pop(old_seq, None)
        if old_count is None:
            continue

        # Subtract all pairs from pair_counts and pair_to_seqs
        if len(old_seq) >= 2:
            for i in range(len(old_seq) - 1):
                p = (old_seq[i], old_seq[i + 1])
                # Decrease count
                pair_counts[p] -= old_count
                if pair_counts[p] <= 0:
                    # Clean up if no occurrences left
                    del pair_counts[p]
                    pair_to_seqs.pop(p, None)
                else:
                    # Remove this sequence from the set
                    seqs_set = pair_to_seqs.get(p)
                    if seqs_set is not None:
                        seqs_set.discard(old_seq)
                        if not seqs_set:
                            pair_to_seqs.pop(p, None)

        # Build merged sequence and aggregate counts
        new_seq = merge_pair_in_sequence(old_seq, most_freq_pair, new_token)
        merged_seqs[new_seq] += old_count

    # 2) Add NEW sequences + their pairs into freq, pair_counts, pair_to_seqs
    for new_seq, added_count in merged_seqs.items():
        # Update corpus
        prev = freq.get(new_seq, 0)
        freq[new_seq] = prev + added_count

        if len(new_seq) < 2:
            continue

        for i in range(len(new_seq) - 1):
            p = (new_seq[i], new_seq[i + 1])
            pair_counts[p] = pair_counts.get(p, 0) + added_count
            # If seq was already in the set due to prev count, set.add handles it
            if p not in pair_to_seqs:
                pair_to_seqs[p] = set()
            pair_to_seqs[p].add(new_seq)

    return new_token


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    desired_num_chunks: int,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """BPE training.
    Each iteration:
    - recomputes pair_counts over entire corpus
    - pick most_freq_pair
    - merge that pair in every repr to give new freq
    """
    if vocab_size <= 256:
        raise ValueError("vocab_size must be > 256 (need room for merges + specials).")
    
    # Parallel pretokenization
    freq: dict[tuple[bytes, ...], int] = pretokenize_file(
        input_path=input_path,
        desired_num_chunks=desired_num_chunks,
        special_tokens=special_tokens
    )

    # Init vocab and merges
    vocab: dict[int, bytes] = build_initial_vocab()
    merges: list[tuple[bytes, bytes]] = []
    next_id = len(vocab)

    # since we need to add special tokens to vocab after merging
    # and each merge gives exactly one more new token
    # this guarantees the max_merges so vocab does not exceed vocab_size
    max_merges = vocab_size - len(vocab) - len(special_tokens)
    if max_merges < 0:
        raise ValueError(
            f"vocab_size={vocab_size} too small for {len(vocab)} bytes + {len(special_tokens)} specials"
        )

    # Build initial stats once
    pair_counts, pair_to_seqs = build_pair_stats(freq)
    
    # Main BPE loop
    for _ in range(max_merges):
        if not pair_counts:
            break

        # Pick best pair (freq + lexicographic tie-breaking)
        most_freq_pair, most_freq_count = find_most_freq_pair(pair_counts)
        if most_freq_pair is None or most_freq_count < 1:
            break

        # Apply merge to whole corpus
        # new_token = apply_merge(freq, most_freq_pair)
        new_token = apply_merge_incremental(freq, most_freq_pair, pair_counts, pair_to_seqs)

        # Record merge and add new token to vocab
        merges.append(most_freq_pair)
        vocab[next_id] = new_token
        next_id += 1

    # Add special tokens
    for special_token in special_tokens:
        vocab[next_id] = special_token.encode("utf-8")
        next_id += 1

    assert len(vocab) <= vocab_size, f"Final vocab {len(vocab)} > vocab_size {vocab_size}"
    return vocab, merges
