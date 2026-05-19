# pipeline-service/tests/test_detect_communities.py
import itertools

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from pipeline.detect_communities import detect_communities


def _make_adjacency(n, edges):
    """Build a symmetric sparse adjacency matrix from (i, j) pairs."""
    arr = np.zeros((n, n), dtype=np.float64)
    for i, j in edges:
        arr[i, j] = 1.0
        arr[j, i] = 1.0
    return csr_matrix(arr)


def _clique(nodes):
    """All undirected edges among a set of nodes."""
    return list(itertools.combinations(nodes, 2))


def _pop(n):
    return [f"u{i}" for i in range(n)]


def _idx(user_id):
    return int(user_id[1:])


def test_well_separated_communities():
    edges = _clique([0, 1, 2, 3]) + _clique([4, 5, 6, 7])
    adjacency = _make_adjacency(8, edges)
    communities, _, _ = detect_communities(adjacency, _pop(8), 3, 25)

    assert len(communities) >= 2
    # No community mixes users from the two cliques.
    for community in communities:
        indices = {_idx(u) for u in community["userIds"]}
        assert indices <= {0, 1, 2, 3} or indices <= {4, 5, 6, 7}


def test_deterministic():
    edges = _clique([0, 1, 2, 3]) + _clique([4, 5, 6, 7])
    adjacency = _make_adjacency(8, edges)

    first, _, _ = detect_communities(adjacency, _pop(8), 3, 25)
    second, _, _ = detect_communities(adjacency, _pop(8), 3, 25)

    assert first == second


def test_min_group_size_suppression():
    edges = [(0, 1)] + _clique([2, 3, 4, 5])
    adjacency = _make_adjacency(6, edges)
    communities, suppressed, _ = detect_communities(adjacency, _pop(6), 3, 25)

    # The 2-user clique {0,1} is below the minimum group size.
    size_suppressed = [s for s in suppressed if "minimum group size" in s["reason"]]
    assert len(size_suppressed) >= 1
    # The 4-user clique survives.
    assert any(len(c["userIds"]) == 4 for c in communities)


def test_max_roles_gate():
    edges = (_clique([0, 1, 2, 3]) + _clique([4, 5, 6, 7])
             + _clique([8, 9, 10, 11]) + _clique([12, 13, 14, 15]))
    adjacency = _make_adjacency(16, edges)
    communities, suppressed, _ = detect_communities(adjacency, _pop(16), 3, 2)

    assert len(communities) == 2
    max_roles_suppressed = [s for s in suppressed if "maxRoles" in s["reason"]]
    assert len(max_roles_suppressed) == 2


def test_all_singletons():
    adjacency = _make_adjacency(5, [])
    communities, _, _ = detect_communities(adjacency, _pop(5), 3, 25)

    assert len(communities) == 0


def test_communities_internally_connected():
    edges = _clique([0, 1, 2, 3]) + _clique([4, 5, 6, 7])
    adjacency = _make_adjacency(8, edges)
    communities, _, _ = detect_communities(adjacency, _pop(8), 3, 25)

    for community in communities:
        members = [_idx(u) for u in community["userIds"]]
        subgraph = adjacency[members, :][:, members]
        n_components, _ = connected_components(subgraph, directed=False)
        assert n_components == 1


def test_suppressed_recorded():
    edges = [(0, 1)] + _clique([2, 3, 4, 5])
    adjacency = _make_adjacency(6, edges)
    _, suppressed, _ = detect_communities(adjacency, _pop(6), 3, 25)

    assert len(suppressed) >= 1
    for entry in suppressed:
        assert isinstance(entry["communityId"], int)
        assert isinstance(entry["size"], int)
        assert isinstance(entry["reason"], str)


def test_modularity_returned():
    edges = _clique([0, 1, 2, 3]) + _clique([4, 5, 6, 7])
    adjacency = _make_adjacency(8, edges)
    _, _, modularity = detect_communities(adjacency, _pop(8), 3, 25)

    assert isinstance(modularity, float)
