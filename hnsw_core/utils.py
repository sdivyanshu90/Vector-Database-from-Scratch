from __future__ import annotations

from typing import List, Sequence

import numpy as np


def now_ns() -> int:
    return int(np.datetime64("now", "ns").astype(np.int64))


def chunk_ids(values: Sequence[int], chunk_size: int) -> List[List[int]]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    return [list(values[index:index + chunk_size]) for index in range(0, len(values), chunk_size)]