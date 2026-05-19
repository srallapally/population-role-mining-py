# pipeline-service/tests/test_build_similarity_graph.py
import numpy as np
import pytest
from scipy.sparse import csr_matrix

from pipeline.build_similarity_graph import build_similarity_graph


def _make_residual(n_users, n_ents, assignments):
    """Build a binary CSR matrix from a list of (user_idx, ent_idx) pairs."""
    arr = np.zeros((n_users, n_ents), dtype=np.float32)
    for user_idx, ent_idx in assignments:
        arr[user_idx, ent_idx] = 1.0
    return csr_matrix(arr)


def test_basic_jaccard():
    # User 0: {0,1,2}  User 1: {0,1,3}  User 2: {4}  User 3: {0,1,2}
    residual = _make_residual(4, 5, [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 3),
        (2, 4),
        (3, 0), (3, 1), (3, 2),
    ])
    adjacency, _ = build_similarity_graph(residual, 0.0)
    a = adjacency.toarray()

    assert a[0][1] == pytest.approx(0.5)   # 2/4
    assert a[0][3] == pytest.approx(1.0)   # 3/3
    assert a[1][3] == pytest.approx(0.5)   # 2/4
    assert a[0][2] == 0.0
    assert a[1][2] == 0.0
    assert a[2][3] == 0.0


def test_zero_both_empty():
    residual = _make_residual(2, 3, [])
    adjacency, metrics = build_similarity_graph(residual, 0.0)

    assert adjacency.nnz == 0
    assert metrics["edgeCount"] == 0


def test_zero_one_empty():
    # User 0 holds entitlements; user 1 holds none.
    residual = _make_residual(2, 3, [(0, 0), (0, 1)])
    adjacency, metrics = build_similarity_graph(residual, 0.0)

    assert adjacency.nnz == 0
    assert metrics["edgeCount"] == 0


def test_threshold_applied():
    # User 0: {0,1,2}  User 1: {0,3}  -> J = 1/4 = 0.25
    residual = _make_residual(2, 4, [(0, 0), (0, 1), (0, 2), (1, 0), (1, 3)])
    adjacency, metrics = build_similarity_graph(residual, 0.30)

    assert adjacency.nnz == 0
    assert metrics["edgeCount"] == 0


def test_threshold_boundary():
    # User 0: {0}  User 1: {0,1}  -> J = 1/2 = 0.50, kept at threshold 0.50
    residual = _make_residual(2, 2, [(0, 0), (1, 0), (1, 1)])
    adjacency, metrics = build_similarity_graph(residual, 0.50)

    assert metrics["edgeCount"] == 1
    assert adjacency.toarray()[0][1] == pytest.approx(0.5)


def test_singletons_counted():
    # Users 0-2 share {0,1}; users 3 and 4 hold unique entitlements.
    residual = _make_residual(5, 4, [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 2),
        (4, 3),
    ])
    _, metrics = build_similarity_graph(residual, 0.30)

    assert metrics["singletonCount"] == 2


def test_edge_count():
    # Users 0-2 form a triangle; user 3 is a singleton.
    residual = _make_residual(4, 3, [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 2),
    ])
    _, metrics = build_similarity_graph(residual, 0.30)

    assert metrics["edgeCount"] == 3


def test_graph_density():
    # Same 4 users, 3 edges. max_edges = 4*3/2 = 6 -> density 0.5.
    residual = _make_residual(4, 3, [
        (0, 0), (0, 1),
        (1, 0), (1, 1),
        (2, 0), (2, 1),
        (3, 2),
    ])
    _, metrics = build_similarity_graph(residual, 0.30)

    assert metrics["graphDensity"] == 0.5


def test_largest_connected_component():
    # Chain 0-1-2-3 (consecutive overlaps only); users 4 and 5 isolated.
    residual = _make_residual(6, 5, [
        (0, 0), (0, 1),
        (1, 1), (1, 2),
        (2, 2), (2, 3),
        (3, 3), (3, 4),
    ])
    _, metrics = build_similarity_graph(residual, 0.30)

    assert metrics["largestConnectedComponentPct"] == round(4 / 6, 4)


def test_symmetric():
    residual = _make_residual(4, 5, [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 3),
        (2, 4),
        (3, 0), (3, 1), (3, 2),
    ])
    adjacency, _ = build_similarity_graph(residual, 0.0)

    assert (adjacency - adjacency.T).nnz == 0


def test_no_self_loops():
    residual = _make_residual(4, 5, [
        (0, 0), (0, 1), (0, 2),
        (1, 0), (1, 1), (1, 3),
        (2, 4),
        (3, 0), (3, 1), (3, 2),
    ])
    adjacency, _ = build_similarity_graph(residual, 0.0)

    assert adjacency.diagonal().sum() == 0
