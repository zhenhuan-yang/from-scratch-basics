import os
import time
import logging
from collections import Counter
from pretokenization import pretokenize_file_in_parallel


logger = logging.getLogger(__name__)


def count_pairs(freq: dict[tuple[bytes, ...], int]) -> Counter[tuple[bytes, bytes]]:
    """Count adjacent token pairs across all words, weighted by word frequency.
    """
    return