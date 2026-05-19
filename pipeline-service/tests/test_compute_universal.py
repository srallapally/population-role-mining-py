# pipeline-service/tests/test_compute_universal.py
import copy

import numpy as np
from scipy.sparse import csr_matrix

from pipeline.compute_universal import compute_universal
from tests.conftest import DEFAULT_TEST_PARAMS

SESSION_ID = "test-session"
POPULATION_SUMMARY = {"filterDescription": "test", "filterCriteria": {}}


def _params(**overrides):
    """A resolved-parameters dict based on DEFAULT_TEST_PARAMS."""
    params = copy.deepcopy(DEFAULT_TEST_PARAMS)
    params.update(overrides)
    return params


def _make_matrix(holders, population_ids, ent_ids):
    """Build a binary CSR matrix and entitlement_index from known data.

    holders: {ent_id: iterable of user row-indices that hold it}.
    Columns follow lexicographically sorted ent_ids — matching
    build_sparse_matrix — so this avoids depending on it.
    """
    ent_ids = sorted(ent_ids)
    entitlement_index = {e: i for i, e in enumerate(ent_ids)}
    arr = np.zeros((len(population_ids), len(ent_ids)), dtype=np.float32)
    for ent_id, rows in holders.items():
        col = entitlement_index[ent_id]
        for row in rows:
            arr[row, col] = 1.0
    return csr_matrix(arr), entitlement_index


def _pop(n):
    return [f"u{i}" for i in range(n)]


def _role_ent_ids(role):
    return {m["entId"] for m in role["entitlementMetadata"]}


def test_binary_split():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {
            "ent-a": range(5),   # 5/5
            "ent-b": range(5),   # 5/5
            "ent-c": range(3),   # 3/5
            "ent-d": range(2),   # 2/5
            "ent-e": range(1),   # 1/5
        },
        population_ids,
        ["ent-a", "ent-b", "ent-c", "ent-d", "ent-e"],
    )
    roles, residual, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90), POPULATION_SUMMARY, {},
    )

    universal_ents = set()
    for role in roles:
        universal_ents |= _role_ent_ids(role)
    assert universal_ents == {"ent-a", "ent-b"}
    # Universal columns are zeroed in the residual; the rest survive.
    assert residual[:, ei["ent-a"]].nnz == 0
    assert residual[:, ei["ent-b"]].nnz == 0
    assert residual[:, ei["ent-c"]].nnz == 3
    assert residual[:, ei["ent-d"]].nnz == 2
    assert residual[:, ei["ent-e"]].nnz == 1


def test_no_universals():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {"ent-a": range(3), "ent-b": range(3), "ent-c": range(3)},  # each 3/5
        population_ids,
        ["ent-a", "ent-b", "ent-c"],
    )
    roles, residual, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90), POPULATION_SUMMARY, {},
    )

    assert roles == []
    assert residual.toarray().tolist() == matrix.toarray().tolist()


def test_single_birthright_cluster():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {"ent-a": range(5), "ent-b": range(5), "ent-c": range(5)},
        population_ids,
        ["ent-a", "ent-b", "ent-c"],
    )
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90, birthrightCooccurrenceThreshold=0.95),
        POPULATION_SUMMARY, {},
    )

    assert len(roles) == 1
    assert roles[0]["roleType"] == "layer1_universal"
    assert roles[0]["entitlementCount"] == 3
    assert len(roles[0]["entitlementMetadata"]) == 3


def test_multiple_birthright_clusters():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {
            "ent-a": {0, 1, 2, 3, 4},
            "ent-b": {0, 1, 2, 3, 4},
            "ent-c": {0, 1, 2, 3},   # J(C, A) = 4/5 = 0.80 < 0.95
        },
        population_ids,
        ["ent-a", "ent-b", "ent-c"],
    )
    # universalThreshold 0.80 so ent-c (prevalence 0.80) is also universal.
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.80, birthrightCooccurrenceThreshold=0.95),
        POPULATION_SUMMARY, {},
    )

    assert len(roles) == 2
    clusters = sorted((_role_ent_ids(r) for r in roles), key=len)
    assert clusters == [{"ent-c"}, {"ent-a", "ent-b"}]


def test_member_count_is_intersection():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {
            "ent-a": {0, 1, 2, 3},
            "ent-b": {0, 1, 2, 4},
        },
        population_ids,
        ["ent-a", "ent-b"],
    )
    # Low cooccurrence threshold so A and B land in one cluster.
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.80, birthrightCooccurrenceThreshold=0.50),
        POPULATION_SUMMARY, {},
    )

    assert len(roles) == 1
    # memberCount is the intersection of holder sets: {0, 1, 2}.
    assert roles[0]["memberCount"] == 3


def test_entitlement_ordering():
    population_ids = _pop(100)
    # Nested holder sets so all three cluster together; prevalences
    # 0.95 / 0.98 / 0.95 with ent-y the highest.
    matrix, ei = _make_matrix(
        {
            "ent-a": range(95),                  # 0.95
            "ent-y": range(98),                  # 0.98
            "ent-b": set(range(94)) | {95},      # 0.95
        },
        population_ids,
        ["ent-a", "ent-b", "ent-y"],
    )
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90, birthrightCooccurrenceThreshold=0.95),
        POPULATION_SUMMARY, {},
    )

    assert len(roles) == 1
    order = [m["entId"] for m in roles[0]["entitlementMetadata"]]
    # 0.98 first, then the two 0.95 entries ascending by entId.
    assert order == ["ent-y", "ent-a", "ent-b"]


def test_entitlement_enrichment(ent_lookup):
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {"ent-ad-domain-users": range(5)},
        population_ids,
        ["ent-ad-domain-users"],
    )
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90), POPULATION_SUMMARY, ent_lookup,
    )

    assert len(roles) == 1
    meta = roles[0]["entitlementMetadata"][0]
    expected = ent_lookup["ent-ad-domain-users"]
    # Enriched display fields come from ent_lookup, not the raw ID.
    assert meta["displayName"] == expected["displayName"]
    assert meta["displayName"] != "ent-ad-domain-users"
    assert meta["appName"] == "Active Directory"


def test_residual_matrix():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {
            "ent-a": range(5),   # universal -> col 0
            "ent-b": range(5),   # universal -> col 1
            "ent-c": range(2),
            "ent-d": range(1),
        },
        population_ids,
        ["ent-a", "ent-b", "ent-c", "ent-d"],
    )
    _, residual, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90), POPULATION_SUMMARY, {},
    )

    assert residual[:, ei["ent-a"]].nnz == 0
    assert residual[:, ei["ent-b"]].nnz == 0
    # Non-universal columns are unchanged.
    assert residual[:, ei["ent-c"]].nnz == matrix[:, ei["ent-c"]].nnz
    assert residual[:, ei["ent-d"]].nnz == matrix[:, ei["ent-d"]].nnz


def test_high_residual_prevalence():
    population_ids = _pop(20)
    matrix, ei = _make_matrix(
        {"ent-mid": range(17)},   # 17/20 = 0.85
        population_ids,
        ["ent-mid"],
    )
    _, _, high_residual = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(coverageThreshold=0.80, universalThreshold=0.90),
        POPULATION_SUMMARY, {},
    )

    entries = {e["entId"]: e for e in high_residual}
    assert "ent-mid" in entries
    assert entries["ent-mid"]["prevalenceInPop"] == 0.85


def test_high_residual_excludes_universal():
    population_ids = _pop(25)
    matrix, ei = _make_matrix(
        {"ent-x": range(23)},   # 23/25 = 0.92, above universalThreshold
        population_ids,
        ["ent-x"],
    )
    _, _, high_residual = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(coverageThreshold=0.80, universalThreshold=0.90),
        POPULATION_SUMMARY, {},
    )

    assert "ent-x" not in {e["entId"] for e in high_residual}


def test_justification_metadata_layer1():
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {"ent-a": range(5)}, population_ids, ["ent-a"]
    )
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90), POPULATION_SUMMARY, {},
    )

    assert len(roles) == 1
    jm = roles[0]["justificationMetadata"]
    for key in ("sessionId", "filterDescription", "filterCriteria",
                "layer", "populationSize", "thresholds"):
        assert key in jm
    assert jm["layer"] == "layer1"
    for key in ("communityId", "communityModularity",
                "cohesionInterpretation", "outliers"):
        assert key not in jm


def test_role_naming(ent_lookup):
    population_ids = _pop(5)
    matrix, ei = _make_matrix(
        {
            "ent-ad-domain-users": range(5),
            "ent-vpn-allusers": range(5),
        },
        population_ids,
        ["ent-ad-domain-users", "ent-vpn-allusers"],
    )
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90, birthrightCooccurrenceThreshold=0.95),
        POPULATION_SUMMARY, ent_lookup,
    )

    assert len(roles) == 1
    name = roles[0]["name"]
    assert "Active Directory" in name
    assert "Pulse Secure VPN" in name
    assert "Birthright" in name
    assert f"(n={roles[0]['memberCount']})" in name


def test_complete_linkage_not_transitive():
    # Holder sets: J(A,B) and J(B,C) >= 0.95 but J(A,C) < 0.95.
    # (Jaccard distance is a metric, so J(A,C) cannot be as low as the
    # spec's literal 0.55 when the other two are ~0.96 — it is bounded
    # below by ~0.92. 0.92 < 0.95 still demonstrates non-transitivity.)
    a = set(range(0, 94)) | {96, 97}            # 96 holders
    b = set(range(0, 96))                       # 96 holders
    c = set(range(2, 96)) | {100, 101}          # 96 holders
    # J(A,B) = J(B,C) = 0.959 ; J(A,C) = 0.920
    population_ids = _pop(102)
    matrix, ei = _make_matrix(
        {"ent-a": a, "ent-b": b, "ent-c": c},
        population_ids,
        ["ent-a", "ent-b", "ent-c"],
    )
    roles, _, _ = compute_universal(
        matrix, ei, population_ids, SESSION_ID,
        _params(universalThreshold=0.90, birthrightCooccurrenceThreshold=0.95),
        POPULATION_SUMMARY, {},
    )

    # Complete linkage keeps {A,B} and {C} apart — not one cluster of 3.
    assert len(roles) == 2
    clusters = sorted((_role_ent_ids(r) for r in roles), key=len)
    assert clusters == [{"ent-c"}, {"ent-a", "ent-b"}]
