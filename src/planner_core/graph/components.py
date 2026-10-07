"""Strongly connected components.

Without this, snapping onto a mis-tagged isolated fragment — a driveway stub,
a path whose only link is a `bicycle=no` way — yields "no route found" with no
explanation. Recording each node's component makes that failure detectable
before a search is even started, and it is the difference between a confusing
error and an actionable one.

Strong (not weak) connectivity is what matters: a one-way trail you can enter
but never leave is a dead end, and weak connectivity would not notice.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from planner_core.config.logging import get_logger

logger = get_logger(__name__)


def strongly_connected(
    indptr: npt.NDArray[np.int32],
    arc_head: npt.NDArray[np.int32],
    arc_allowed: npt.NDArray[np.uint8],
    n_nodes: int,
) -> tuple[npt.NDArray[np.int32], int]:
    """Label every node with its strongly connected component.

    Returns the labels and the label of the largest component. Only arcs that
    are actually traversable participate — a disallowed arc is not a
    connection, and treating it as one would hide exactly the dead ends this
    is meant to find.
    """
    if n_nodes == 0:
        return np.zeros(0, dtype=np.int32), 0

    matrix = csr_matrix(
        (arc_allowed.astype(np.int8), arc_head.astype(np.int32), indptr.astype(np.int32)),
        shape=(n_nodes, n_nodes),
    )
    matrix.eliminate_zeros()

    n_components, labels = connected_components(matrix, directed=True, connection="strong")
    sizes = np.bincount(labels, minlength=n_components)
    giant = int(np.argmax(sizes))

    logger.info(
        "Connectivity: %d component(s); largest holds %d/%d nodes (%.1f%%)",
        n_components,
        int(sizes[giant]),
        n_nodes,
        100.0 * sizes[giant] / n_nodes,
    )
    return labels.astype(np.int32), giant
