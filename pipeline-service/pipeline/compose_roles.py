# pipeline-service/pipeline/compose_roles.py
import uuid

import numpy as np
from scipy.sparse import csr_matrix


def compose_roles(
    communities: list[dict],
    residual: csr_matrix,
    entitlement_index: dict[str, int],
    population_ids: list[str],
    modularity: float,
    session_id: str,
    parameters: dict,
    population_summary: dict,
    ent_lookup: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """
    For each community: compute entitlement prevalence, two-tier tiering,
    cohesion, app breakdown, outlier analysis.

    Returns:
      layer2_roles: list of role documents
      suppressed: communities suppressed by role-defining entitlement gate
    """
    coverage_threshold = parameters["coverageThreshold"]
    soft_threshold = parameters["softThreshold"]
    outlier_threshold = parameters["outlierThreshold"]

    idx_to_ent = {v: k for k, v in entitlement_index.items()}
    user_index = {uid: idx for idx, uid in enumerate(population_ids)}

    layer2_roles = []
    suppressed = []

    for community in communities:
        community_id = community["communityId"]
        user_ids = community["userIds"]
        member_indices = [user_index[uid] for uid in user_ids if uid in user_index]

        if not member_indices:
            continue

        community_matrix = residual[member_indices, :]
        n_members = len(member_indices)

        col_sums = np.asarray(community_matrix.sum(axis=0)).flatten()
        prevalence = col_sums / n_members

        role_defining_cols = [i for i, p in enumerate(prevalence) if p >= coverage_threshold]

        if not role_defining_cols:
            suppressed.append({
                "communityId": community_id,
                "size": n_members,
                "reason": "no entitlement met role-defining coverage threshold",
            })
            continue

        common_not_universal_cols = [
            i for i, p in enumerate(prevalence)
            if soft_threshold <= p < coverage_threshold
        ]

        role_id = str(uuid.uuid4())
        role_defining_ent_ids = [idx_to_ent[c] for c in role_defining_cols]

        ent_meta = _build_ent_meta(
            role_defining_cols, common_not_universal_cols,
            prevalence, idx_to_ent, ent_lookup
        )

        cohesion, cohesion_interpretation = _compute_cohesion(community_matrix)
        applications = _build_app_breakdown(
            [em for em in ent_meta if em["tier"] == "role_defining"], ent_lookup
        )

        outliers = _compute_outliers(
            member_indices, user_ids, residual, role_defining_cols,
            outlier_threshold, idx_to_ent
        )

        layer2_roles.append({
            "id": role_id,
            "status": "candidate",
            "roleType": "layer2_candidate",
            "sessionId": session_id,
            "name": f"Community {community_id} (n={n_members})",
            "memberCount": n_members,
            "entitlementCount": len(role_defining_ent_ids),
            "entitlements": " ".join(
                em["entId"] for em in ent_meta if em["tier"] == "role_defining"
            ),
            "confidence": round(cohesion, 4),
            "memberIds": user_ids,
            "applications": applications,
            "entitlementMetadata": ent_meta,
            "justificationMetadata": {
                "sessionId": session_id,
                "filterDescription": population_summary.get("filterDescription", ""),
                "filterCriteria": parameters["filterCriteria"],
                "layer": "layer2",
                "populationSize": len(population_ids),
                "thresholds": _thresholds(parameters),
                "communityId": community_id,
                "communityModularity": round(modularity, 4),
                "cohesionInterpretation": cohesion_interpretation,
                "outliers": outliers,
            },
        })

    # Canonical sort: confidence desc, memberCount desc, name asc
    layer2_roles.sort(
        key=lambda r: (-r["confidence"], -r["memberCount"], r["name"])
    )

    return layer2_roles, suppressed


def _build_ent_meta(
    role_defining_cols, common_not_universal_cols,
    prevalence, idx_to_ent, ent_lookup
) -> list[dict]:
    meta = []
    for cols, tier in [(role_defining_cols, "role_defining"),
                       (common_not_universal_cols, "common_not_universal")]:
        tier_meta = []
        for c in cols:
            ent_id = idx_to_ent[c]
            info = ent_lookup.get(ent_id, {})
            tier_meta.append({
                "entId": ent_id,
                "displayName": info.get("displayName", ent_id),
                "appId": info.get("appId", ""),
                "appName": info.get("appName", ""),
                "entitlementType": info.get("entitlementType", ""),
                "criticality": "",
                "prevalenceInRole": round(float(prevalence[c]), 4),
                "tier": tier,
            })
        tier_meta.sort(key=lambda e: (-e["prevalenceInRole"], e["entId"]))
        meta.extend(tier_meta)
    return meta


def _compute_cohesion(community_matrix: csr_matrix) -> tuple[float, str]:
    n = community_matrix.shape[0]
    if n < 2:
        score = 1.0
    else:
        row_sums = np.asarray(community_matrix.sum(axis=1)).flatten().astype(np.float64)

        # Intersection counts — sparse
        intersection = (community_matrix @ community_matrix.T).astype(np.float64).tocsr()
        intersection.eliminate_zeros()

        rows, cols = intersection.nonzero()
        total = 0.0
        pair_count = 0

        for k in range(len(rows)):
            i, j = rows[k], cols[k]
            if i >= j:  # upper triangle only, skip diagonal
                continue
            union = row_sums[i] + row_sums[j] - intersection[i, j]
            total += intersection[i, j] / union if union > 0 else 0.0
            pair_count += 1

        # Add zero-contribution pairs (no shared entitlements, not in nonzero)
        all_pairs = n * (n - 1) // 2
        score = total / all_pairs if all_pairs > 0 else 0.0

    if score >= 0.7:
        interpretation = f"High (score={score:.2f}, threshold>=0.7)"
    elif score >= 0.4:
        interpretation = f"Medium (score={score:.2f}, threshold 0.4–0.7)"
    else:
        interpretation = f"Low (score={score:.2f}, threshold<0.4)"

    return score, interpretation


def _build_app_breakdown(role_defining_meta: list[dict], ent_lookup: dict) -> list[dict]:
    app_counts: dict[str, str] = {}
    for em in role_defining_meta:
        app_counts[em["appName"]] = app_counts.get(em["appName"], 0) + 1
    total = len(role_defining_meta)
    return [
        {"appName": app, "appId": "", "entitlementCount": cnt,
         "pctOfRole": round(cnt / total, 4) if total else 0}
        for app, cnt in app_counts.items()
    ]


def _compute_outliers(
    member_indices: list[int],
    user_ids: list[str],
    residual: csr_matrix,
    role_defining_cols: list[int],
    outlier_threshold: float,
    idx_to_ent: dict[int, str],
) -> list[dict]:
    role_defining_set = set(role_defining_cols)
    outliers = []

    for idx, uid in zip(member_indices, user_ids):
        user_ents = set(residual[idx].nonzero()[1])
        missing = role_defining_set - user_ents
        extra = user_ents - role_defining_set

        under_score = len(missing) / len(role_defining_set) if role_defining_set else 0.0
        over_score = len(extra) / len(user_ents) if user_ents else 0.0

        if under_score > outlier_threshold or over_score > outlier_threshold:
            n_role = len(role_defining_set)
            outliers.append({
                "userId": uid,
                "underProvisioningScore": round(under_score, 4),
                "overProvisioningScore": round(over_score, 4),
                "underProvisioned": [idx_to_ent[c] for c in sorted(missing)],
                "overProvisioned": [idx_to_ent[c] for c in sorted(extra)],
                "plainLanguage": (
                    f"Missing {len(missing)} of {n_role} role-defining entitlements "
                    f"({under_score:.0%}). "
                    f"Holds {len(extra)} extra entitlements outside the role definition "
                    f"({over_score:.0%} of residual access is non-role)."
                ),
            })

    return outliers


def _thresholds(parameters: dict) -> dict:
    return {
        "universalThreshold": parameters["universalThreshold"],
        "coverageThreshold": parameters["coverageThreshold"],
        "softThreshold": parameters["softThreshold"],
        "similarityThreshold": parameters["similarityThreshold"],
        "minGroupSize": parameters["minGroupSize"],
        "noiseFilterValue": parameters["noiseFilterValue"],
        "outlierThreshold": parameters["outlierThreshold"],
        "birthrightCooccurrenceThreshold": parameters["birthrightCooccurrenceThreshold"],
    }
