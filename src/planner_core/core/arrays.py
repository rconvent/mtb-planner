"""Helpers for working with ragged runs stored in flat arrays.

Ways, segments, and edge geometry all share the same shape: one long array
plus offsets marking where each run begins. Selecting a subset of those runs
is common enough, and slow enough done in Python, to be worth a helper.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def ragged_indices(
    starts: npt.NDArray[np.int64], lengths: npt.NDArray[np.int64]
) -> npt.NDArray[np.int64]:
    """Element indices selecting a set of variable-length runs.

    Given the start and length of each wanted run, produce the concatenated
    indices of their elements. The alternative — a Python loop building one
    `arange` per run — costs seconds once there are millions of runs.
    """
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    output_starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.repeat(starts - output_starts, lengths) + np.arange(total, dtype=np.int64)
