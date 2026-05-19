# pipeline-service/tests/test_build_sparse_matrix.py
from pipeline.build_sparse_matrix import build_sparse_matrix


def test_basic_matrix():
    population_ids = ["u1", "u2", "u3", "u4"]
    population_assignments = {
        "u1": ["ent-a", "ent-b"],
        "u2": ["ent-b", "ent-c"],
        "u3": ["ent-c", "ent-d"],
        "u4": ["ent-d", "ent-e"],
    }
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=1
    )

    assert matrix.shape == (4, 5)
    assert dropped_count == 0
    # Columns are the lexicographically sorted entitlement IDs.
    assert entitlement_index == {
        "ent-a": 0, "ent-b": 1, "ent-c": 2, "ent-d": 3, "ent-e": 4,
    }
    assert matrix.toarray().tolist() == [
        [1, 1, 0, 0, 0],  # u1: a, b
        [0, 1, 1, 0, 0],  # u2: b, c
        [0, 0, 1, 1, 0],  # u3: c, d
        [0, 0, 0, 1, 1],  # u4: d, e
    ]


def test_deterministic_ordering():
    population_ids = ["u1", "u2", "u3"]
    forward = {
        "u1": ["ent-b", "ent-a"],
        "u2": ["ent-a", "ent-c"],
        "u3": ["ent-c", "ent-b"],
    }
    reversed_order = {k: forward[k] for k in reversed(list(forward))}

    m1, idx1, dropped1 = build_sparse_matrix(population_ids, forward, 1)
    m2, idx2, dropped2 = build_sparse_matrix(population_ids, reversed_order, 1)

    assert idx1 == idx2
    assert dropped1 == dropped2
    assert m1.toarray().tolist() == m2.toarray().tolist()


def test_noise_filter_drops():
    population_ids = ["u1", "u2", "u3", "u4"]
    population_assignments = {
        "u1": ["ent-common", "ent-rare"],
        "u2": ["ent-common"],
        "u3": ["ent-common"],
        "u4": ["ent-common"],
    }
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=2
    )

    # ent-rare is held by only 1 user, below the noise threshold.
    assert "ent-rare" not in entitlement_index
    assert dropped_count >= 1


def test_noise_filter_keeps():
    population_ids = ["u1", "u2", "u3", "u4"]
    population_assignments = {
        "u1": ["ent-x"],
        "u2": ["ent-x"],
        "u3": [],
        "u4": [],
    }
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=2
    )

    # ent-x is held by exactly 2 users, meeting the noise threshold.
    assert "ent-x" in entitlement_index


def test_noise_filter_count():
    population_ids = ["u1", "u2", "u3", "u4"]
    common = [f"ent-common-{i}" for i in range(6)]   # 6 entitlements, 3 holders each
    population_assignments = {
        "u1": common + ["ent-rare-1"],
        "u2": common + ["ent-rare-2"],
        "u3": common + ["ent-rare-3"],
        "u4": ["ent-rare-4"],
    }
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=2
    )

    # 10 entitlements total: 6 survive (3 holders), 4 rare ones drop (1 holder).
    assert dropped_count == 4
    assert len(entitlement_index) == 6


def test_empty_assignments():
    population_ids = ["u1", "u2"]
    population_assignments = {
        "u1": ["ent-a"],
        "u2": [],
    }
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=1
    )

    # u2 holds nothing — its row sums to 0.
    u2_row = matrix.toarray()[population_ids.index("u2")]
    assert u2_row.sum() == 0


def test_binary_dedup():
    population_ids = ["u1"]
    population_assignments = {"u1": ["ent-a", "ent-a"]}
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=1
    )

    # Duplicate grant collapses to a single binary 1.0, not 2.0.
    assert matrix.toarray()[0][entitlement_index["ent-a"]] == 1.0
    assert matrix.sum() == 1.0


def test_entitlement_index_sorted():
    population_ids = ["u1"]
    population_assignments = {"u1": ["ent-z", "ent-a", "ent-m"]}
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=1
    )

    # Column indices follow lexicographic order of entitlement IDs.
    assert entitlement_index == {"ent-a": 0, "ent-m": 1, "ent-z": 2}
    first_id = sorted(entitlement_index)[0]
    assert entitlement_index[first_id] == 0


def test_full_seed_data(population_data):
    population_ids, population_assignments = population_data
    matrix, entitlement_index, dropped_count = build_sparse_matrix(
        population_ids, population_assignments, noise_filter_value=2
    )

    assert matrix.shape[0] == 20

    # The 5 ent-unique-* entitlements are each held by a single user.
    for n in range(16, 21):
        assert f"ent-unique-{n}" not in entitlement_index

    # dropped_count is 10, not 5: besides the 5 ent-unique-* entitlements,
    # the 5 outlier extras (ent-extra-a1/a2/a3 on usr-003, ent-extra-b1/b2
    # on usr-012) are also single-holder, so they drop at noise=2 too.
    for ent_id in ("ent-extra-a1", "ent-extra-a2", "ent-extra-a3",
                   "ent-extra-b1", "ent-extra-b2"):
        assert ent_id not in entitlement_index
    assert dropped_count == 10
