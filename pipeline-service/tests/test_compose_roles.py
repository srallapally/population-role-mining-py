# pipeline-service/tests/test_compose_roles.py
import copy

import numpy as np
from scipy.sparse import csr_matrix

from pipeline.compose_roles import compose_roles
from tests.conftest import DEFAULT_TEST_PARAMS

SESSION_ID = "test-session"
MODULARITY = 0.65
POPULATION_SUMMARY = {"filterDescription": "test", "filterCriteria": {}}


def _params(**overrides):
    params = copy.deepcopy(DEFAULT_TEST_PARAMS)
    params.update(overrides)
    return params


def _csr(n_users, n_ents, pairs):
    arr = np.zeros((n_users, n_ents), dtype=np.float32)
    for user_idx, ent_idx in pairs:
        arr[user_idx, ent_idx] = 1.0
    return csr_matrix(arr)


def _make_community_data(n_users, n_ents, community_assignments,
                         community_user_indices):
    """Build a residual CSR matrix, generic entitlement_index, population_ids
    and a single-community list for one test case.

    community_assignments: list of (user_idx, ent_idx) pairs.
    """
    residual = _csr(n_users, n_ents, community_assignments)
    entitlement_index = {f"ent-{i}": i for i in range(n_ents)}
    population_ids = [f"u{i}" for i in range(n_users)]
    communities = [{
        "communityId": 0,
        "userIds": [population_ids[i] for i in community_user_indices],
    }]
    return residual, entitlement_index, population_ids, communities


def _compose(residual, entitlement_index, population_ids, communities,
             params, ent_lookup=None):
    return compose_roles(
        communities, residual, entitlement_index, population_ids,
        MODULARITY, SESSION_ID, params, POPULATION_SUMMARY,
        ent_lookup if ent_lookup is not None else {},
    )


def _tiers(role):
    return {m["entId"]: m["tier"] for m in role["entitlementMetadata"]}


def test_two_tier_assignment():
    # 5-user community. ent-0: 5/5, ent-1: 3/5, ent-2: 1/5.
    residual, ei, pop, communities = _make_community_data(
        5, 3,
        [(u, 0) for u in range(5)]
        + [(u, 1) for u in range(3)]
        + [(0, 2)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(coverageThreshold=0.80, softThreshold=0.50))

    assert len(roles) == 1
    assert _tiers(roles[0]) == {
        "ent-0": "role_defining",
        "ent-1": "common_not_universal",
    }


def test_excluded_below_soft():
    # ent-1 held by 2/5 = 0.40, below softThreshold 0.50.
    residual, ei, pop, communities = _make_community_data(
        5, 2,
        [(u, 0) for u in range(5)] + [(0, 1), (1, 1)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(coverageThreshold=0.80, softThreshold=0.50))

    assert len(roles) == 1
    assert "ent-1" not in {m["entId"] for m in roles[0]["entitlementMetadata"]}


def test_role_defining_gate():
    # All entitlements below coverageThreshold -> community suppressed.
    residual, ei, pop, communities = _make_community_data(
        5, 2,
        [(u, 0) for u in range(3)],   # ent-0 at 3/5 = 0.60
        list(range(5)),
    )
    roles, suppressed = _compose(residual, ei, pop, communities,
                                 _params(coverageThreshold=0.80))

    assert roles == []
    assert len(suppressed) == 1
    assert "no entitlement met role-defining coverage threshold" \
        in suppressed[0]["reason"]


def test_entitlement_ordering():
    # role_defining: ent-a 0.95, ent-b 0.90, ent-c 0.90 (entId tie-break).
    # common_not_universal: ent-d 0.70, ent-e 0.60.
    # Columns deliberately not in entId order.
    entitlement_index = {"ent-c": 0, "ent-a": 1, "ent-b": 2,
                         "ent-e": 3, "ent-d": 4}
    counts = [18, 19, 18, 12, 14]   # per column, out of 20
    arr = np.zeros((20, 5), dtype=np.float32)
    for col, count in enumerate(counts):
        for user in range(count):
            arr[user, col] = 1.0
    residual = csr_matrix(arr)
    population_ids = [f"u{i}" for i in range(20)]
    communities = [{"communityId": 0, "userIds": population_ids}]

    roles, _ = _compose(residual, entitlement_index, population_ids,
                        communities,
                        _params(coverageThreshold=0.80, softThreshold=0.50))

    order = [m["entId"] for m in roles[0]["entitlementMetadata"]]
    assert order == ["ent-a", "ent-b", "ent-c", "ent-d", "ent-e"]


def test_cohesion_high():
    # All 5 members hold an identical entitlement set -> cohesion 1.0.
    residual, ei, pop, communities = _make_community_data(
        5, 2,
        [(u, e) for u in range(5) for e in range(2)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities, _params())

    assert roles[0]["confidence"] >= 0.7
    assert "High" in roles[0]["justificationMetadata"]["cohesionInterpretation"]


def test_cohesion_medium():
    # Moderate overlap -> average pairwise Jaccard ~0.52.
    residual, ei, pop, communities = _make_community_data(
        5, 5,
        [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 1), (1, 2),
         (2, 0), (2, 3), (2, 4),
         (3, 0), (3, 3), (3, 4),
         (4, 0), (4, 1), (4, 2)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities, _params())

    assert 0.4 <= roles[0]["confidence"] < 0.7
    assert "Medium" in roles[0]["justificationMetadata"]["cohesionInterpretation"]


def test_cohesion_low():
    # Minimal overlap -> members share only ent-0.
    residual, ei, pop, communities = _make_community_data(
        5, 12,
        [(0, 0), (0, 1), (0, 2),
         (1, 0), (1, 3), (1, 4),
         (2, 0), (2, 5), (2, 6),
         (3, 0), (3, 7), (3, 8),
         (4, 9), (4, 10), (4, 11)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities, _params())

    assert roles[0]["confidence"] < 0.4
    assert "Low" in roles[0]["justificationMetadata"]["cohesionInterpretation"]


def test_app_breakdown(ent_lookup):
    # 3 Epic entitlements + 2 ServiceNow, all role-defining.
    entitlement_index = {
        "ent-epic-amb-orders": 0,
        "ent-epic-amb-notes": 1,
        "ent-epic-amb-schedule": 2,
        "ent-snow-incident": 3,
        "ent-snow-change": 4,
    }
    residual = _csr(5, 5, [(u, e) for u in range(5) for e in range(5)])
    population_ids = [f"u{i}" for i in range(5)]
    communities = [{"communityId": 0, "userIds": population_ids}]

    roles, _ = _compose(residual, entitlement_index, population_ids,
                        communities, _params(), ent_lookup)

    apps = {a["appName"]: a for a in roles[0]["applications"]}
    assert len(apps) == 2
    assert apps["Epic"]["entitlementCount"] == 3
    assert apps["Epic"]["pctOfRole"] == 0.6
    assert apps["ServiceNow"]["entitlementCount"] == 2
    assert apps["ServiceNow"]["pctOfRole"] == 0.4


def test_app_breakdown_nonempty():
    residual, ei, pop, communities = _make_community_data(
        5, 2, [(u, 0) for u in range(5)], list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities, _params())

    for role in roles:
        assert len(role["applications"]) > 0


def test_outlier_over_provisioned():
    # u0 holds 10 entitlements; the role has 5 role-defining ones.
    residual, ei, pop, communities = _make_community_data(
        5, 10,
        [(0, e) for e in range(10)]
        + [(u, e) for u in range(1, 5) for e in range(5)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(outlierThreshold=0.30))

    outliers = {o["userId"]: o for o in roles[0]["justificationMetadata"]["outliers"]}
    assert "u0" in outliers
    assert outliers["u0"]["overProvisioningScore"] == 0.5
    assert len(outliers["u0"]["overProvisioned"]) == 5


def test_outlier_under_provisioned():
    # u9 holds only 4 of 8 role-defining entitlements.
    residual, ei, pop, communities = _make_community_data(
        10, 8,
        [(u, e) for u in range(9) for e in range(8)]
        + [(9, e) for e in range(4)],
        list(range(10)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(outlierThreshold=0.30))

    outliers = {o["userId"]: o for o in roles[0]["justificationMetadata"]["outliers"]}
    assert "u9" in outliers
    assert outliers["u9"]["underProvisioningScore"] == 0.5
    assert len(outliers["u9"]["underProvisioned"]) == 4


def test_outlier_below_threshold():
    # u10 is missing only 1 of 10 role-defining entitlements (0.10).
    residual, ei, pop, communities = _make_community_data(
        11, 10,
        [(u, e) for u in range(10) for e in range(10)]
        + [(10, e) for e in range(9)],
        list(range(11)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(outlierThreshold=0.30))

    outlier_ids = {o["userId"]
                   for o in roles[0]["justificationMetadata"]["outliers"]}
    assert "u10" not in outlier_ids


def test_outlier_plain_language():
    residual, ei, pop, communities = _make_community_data(
        5, 10,
        [(0, e) for e in range(10)]
        + [(u, e) for u in range(1, 5) for e in range(5)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(outlierThreshold=0.30))

    outliers = roles[0]["justificationMetadata"]["outliers"]
    assert outliers
    text = outliers[0]["plainLanguage"]
    assert "Missing" in text
    assert "role-defining" in text
    assert "extra entitlements" in text


def test_outlier_zero_residual():
    # u0 holds no residual entitlements at all.
    residual, ei, pop, communities = _make_community_data(
        5, 2,
        [(u, 0) for u in range(1, 5)],
        list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities,
                        _params(coverageThreshold=0.80, outlierThreshold=0.30))

    outliers = {o["userId"]: o for o in roles[0]["justificationMetadata"]["outliers"]}
    assert outliers["u0"]["overProvisioningScore"] == 0


def test_canonical_sort():
    # Three communities: confidence desc, then memberCount desc.
    #   comm 1: 5 identical members -> confidence 1.0
    #   comm 2: 3 identical members -> confidence 1.0
    #   comm 3: 4 low-cohesion members -> confidence ~0.33
    # Expected order: comm 1, comm 2 (memberCount tie-break), comm 3.
    pairs = (
        [(u, e) for u in range(5) for e in range(2)]        # comm 1: u0-u4
        + [(u, e) for u in range(5, 8) for e in range(2)]   # comm 2: u5-u7
        + [(8, 0), (8, 2), (9, 0), (9, 3),                  # comm 3: u8-u11
           (10, 0), (10, 4), (11, 0), (11, 5)]
    )
    residual = _csr(12, 6, pairs)
    entitlement_index = {f"ent-{i}": i for i in range(6)}
    population_ids = [f"u{i}" for i in range(12)]
    communities = [
        {"communityId": 1, "userIds": [f"u{i}" for i in range(5)]},
        {"communityId": 2, "userIds": [f"u{i}" for i in range(5, 8)]},
        {"communityId": 3, "userIds": [f"u{i}" for i in range(8, 12)]},
    ]

    roles, _ = _compose(residual, entitlement_index, population_ids,
                        communities, _params())

    assert len(roles) == 3
    assert [r["justificationMetadata"]["communityId"] for r in roles] == [1, 2, 3]
    # The canonical sort key holds for the returned order.
    assert roles == sorted(
        roles,
        key=lambda r: (-r["confidence"], -r["memberCount"], r["name"]),
    )


def test_justification_metadata_layer2():
    residual, ei, pop, communities = _make_community_data(
        5, 2, [(u, 0) for u in range(5)], list(range(5)),
    )
    roles, _ = _compose(residual, ei, pop, communities, _params())

    jm = roles[0]["justificationMetadata"]
    for key in ("sessionId", "filterDescription", "filterCriteria",
                "layer", "populationSize", "thresholds",
                "communityId", "communityModularity",
                "cohesionInterpretation", "outliers"):
        assert key in jm
    assert jm["layer"] == "layer2"
    assert isinstance(jm["communityId"], int)
    assert isinstance(jm["communityModularity"], float)
    assert isinstance(jm["cohesionInterpretation"], str)
    assert isinstance(jm["outliers"], list)


def test_role_naming():
    residual, ei, pop, communities = _make_community_data(
        45, 2, [(u, 0) for u in range(45)], list(range(45)),
    )
    communities[0]["communityId"] = 3
    roles, _ = _compose(residual, ei, pop, communities, _params())

    assert roles[0]["name"] == "Community 3 (n=45)"


def test_enrichment_applied(ent_lookup):
    entitlement_index = {"ent-ad-domain-users": 0, "ent-vpn-allusers": 1}
    residual = _csr(5, 2, [(u, e) for u in range(5) for e in range(2)])
    population_ids = [f"u{i}" for i in range(5)]
    communities = [{"communityId": 0, "userIds": population_ids}]

    roles, _ = _compose(residual, entitlement_index, population_ids,
                        communities, _params(), ent_lookup)

    meta = {m["entId"]: m for m in roles[0]["entitlementMetadata"]}
    ad = meta["ent-ad-domain-users"]
    assert ad["displayName"] == ent_lookup["ent-ad-domain-users"]["displayName"]
    assert ad["displayName"] != "ent-ad-domain-users"
    assert ad["appName"] == "Active Directory"
