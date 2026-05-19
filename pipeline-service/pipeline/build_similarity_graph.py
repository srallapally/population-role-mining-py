# backend/pipeline/build_similarity_graph.py

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import connected_components


def build_similarity_graph(
    residual: csr_matrix, similarity_threshold: float
) -> tuple[csr_matrix, dict]:
    n_users = residual.shape[0]

    # Row norms (cardinalities)
    row_sums = np.asarray(residual.sum(axis=1)).flatten().astype(np.float64)

    # Intersection counts — stays sparse
    intersection = (residual @ residual.T).astype(np.float64)

    # Compute Jaccard only for non-zero intersections (sparse traversal)
    intersection = intersection.tocsr()
    intersection.eliminate_zeros()

    rows, cols = intersection.nonzero()
    data = intersection.data.copy()

    for k in range(len(data)):
        i, j = rows[k], cols[k]
        if i == j:
            data[k] = 0.0
            continue
        union = row_sums[i] + row_sums[j] - data[k]
        data[k] = data[k] / union if union > 0 else 0.0

    jaccard = csr_matrix((data, (rows, cols)), shape=(n_users, n_users))

    # Threshold: keep only edges >= similarity_threshold, drop self-loops
    jaccard.data[jaccard.data < similarity_threshold] = 0.0
    jaccard.eliminate_zeros()

    # Zero diagonal (self-loops from the intersection step)
    jaccard = _zero_diagonal(jaccard)

    adjacency = jaccard

    # Graph metrics — all computed on sparse structure
    edge_count = int(adjacency.nnz // 2)
    max_edges = n_users * (n_users - 1) / 2
    graph_density = round(edge_count / max_edges, 6) if max_edges > 0 else 0.0

    degrees = np.asarray(adjacency.sum(axis=1)).flatten()
    singleton_count = int((degrees == 0).sum())

    n_components, labels = connected_components(adjacency, directed=False)
    if n_users > 0:
        component_sizes = np.bincount(labels)
        lcc_pct = round(float(component_sizes.max()) / n_users, 4)
    else:
        lcc_pct = 0.0

    graph_metrics = {
        "edgeCount": edge_count,
        "graphDensity": graph_density,
        "singletonCount": singleton_count,
        "largestConnectedComponentPct": lcc_pct,
    }

    return adjacency, graph_metrics


def _zero_diagonal(m: csr_matrix) -> csr_matrix:
    m = m.tolil()
    m.setdiag(0)
    m = m.tocsr()
    m.eliminate_zeros()
    return m
