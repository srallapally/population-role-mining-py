# pipeline-service/pipeline/compute_universal.py
import uuid

import numpy as np
from scipy.sparse import csr_matrix


def compute_universal(
    matrix: csr_matrix,
    entitlement_index: dict[str, int],
    population_ids: list[str],
    session_id: str,
    parameters: dict,
    population_summary: dict,
    ent_lookup: dict[str, dict],
) -> tuple[list[dict], csr_matrix, list[dict]]:
    """
    Step 3: prevalence scan, birthright clustering, Layer 1 role construction.

    Returns:
      layer1_roles: list of role documents (not yet written)
      residual_matrix: CSR matrix with universal columns zeroed
      high_residual_prevalence: entitlements in [coverageThreshold, universalThreshold)
    """
    universal_threshold = parameters["universalThreshold"]
    coverage_threshold = parameters["coverageThreshold"]
    birthright_cooccurrence_threshold = parameters["birthrightCooccurrenceThreshold"]
    population_size = len(population_ids)

    n_users, n_ents = matrix.shape
    col_sums = np.asarray(matrix.sum(axis=0)).flatten()
    prevalence = col_sums / population_size

    idx_to_ent = {v: k for k, v in entitlement_index.items()}

    universal_cols = [i for i, p in enumerate(prevalence) if p >= universal_threshold]
    universal_ent_ids = [idx_to_ent[i] for i in universal_cols]

    # High residual prevalence diagnostic
    high_residual_prevalence = []
    for i, p in enumerate(prevalence):
        if coverage_threshold <= p < universal_threshold:
            ent_id = idx_to_ent[i]
            meta = ent_lookup.get(ent_id, {})
            high_residual_prevalence.append({
                "entId": ent_id,
                "displayName": meta.get("displayName", ent_id),
                "appName": meta.get("appName", ""),
                "prevalenceInPop": round(float(p), 4),
            })

    # Cluster universal entitlements by holder-set co-occurrence
    clusters = _cluster_universal_entitlements(
        universal_cols, matrix, birthright_cooccurrence_threshold
    )

    layer1_roles = []
    for cluster_cols in clusters:
        role = _build_layer1_role(
            cluster_cols, matrix, entitlement_index, idx_to_ent,
            prevalence, population_ids, population_size, session_id,
            parameters, population_summary, ent_lookup
        )
        layer1_roles.append(role)

    # Residual matrix: zero out universal columns
    residual = matrix.copy().tolil()
    for col in universal_cols:
        residual[:, col] = 0
    residual = residual.tocsr()
    residual.eliminate_zeros()

    return layer1_roles, residual, high_residual_prevalence


def _cluster_universal_entitlements(
    universal_cols: list[int], matrix: csr_matrix, threshold: float
) -> list[list[int]]:
    """
    Complete-linkage agglomerative clustering on universal entitlement holder sets.
    Two entitlements are placed in the same cluster only if the new entitlement's
    Jaccard with every existing member of the cluster meets the threshold.
    This prevents transitivity from merging entitlements that are not mutually similar.
    """
    if not universal_cols:
        return []

    # Precompute all pairwise Jaccard scores
    n = len(universal_cols)
    jaccard = {}
    for a in range(n):
        for b in range(a + 1, n):
            col_a = universal_cols[a]
            col_b = universal_cols[b]
            holders_a = set(matrix[:, col_a].nonzero()[0])
            holders_b = set(matrix[:, col_b].nonzero()[0])
            union = holders_a | holders_b
            score = len(holders_a & holders_b) / len(union) if union else 0.0
            jaccard[(a, b)] = score
            jaccard[(b, a)] = score

    # Complete-linkage: merge two clusters only if every cross-pair meets threshold
    clusters = [[i] for i in range(n)]

    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                if _all_pairs_meet_threshold(clusters[i], clusters[j], jaccard, threshold):
                    clusters[i] = clusters[i] + clusters[j]
                    clusters.pop(j)
                    merged = True
                    break
            if merged:
                break

    return [[universal_cols[idx] for idx in cluster] for cluster in clusters]


def _all_pairs_meet_threshold(
    cluster_a: list[int], cluster_b: list[int], jaccard: dict, threshold: float
) -> bool:
    for a in cluster_a:
        for b in cluster_b:
            if jaccard.get((a, b), 0.0) < threshold:
                return False
    return True


def _build_layer1_role(
    cluster_cols: list[int],
    matrix: csr_matrix,
    entitlement_index: dict[str, int],
    idx_to_ent: dict[int, str],
    prevalence: np.ndarray,
    population_ids: list[str],
    population_size: int,
    session_id: str,
    parameters: dict,
    population_summary: dict,
    ent_lookup: dict,
) -> dict:
    ent_ids = [idx_to_ent[c] for c in cluster_cols]

    # memberCount: users holding ALL entitlements in cluster
    holder_sets = [set(matrix[:, c].nonzero()[0]) for c in cluster_cols]
    intersection = holder_sets[0]
    for s in holder_sets[1:]:
        intersection &= s
    member_count = len(intersection)

    # Entitlement metadata — sorted descending prevalence, ascending ent_id
    ent_meta = []
    for c, ent_id in zip(cluster_cols, ent_ids):
        meta = ent_lookup.get(ent_id, {})
        ent_meta.append({
            "entId": ent_id,
            "displayName": meta.get("displayName", ent_id),
            "appId": meta.get("appId", ""),
            "appName": meta.get("appName", ""),
            "entitlementType": meta.get("entitlementType", ""),
            "criticality": "",
            "prevalenceInRole": round(float(prevalence[c]), 4),
            "tier": "universal",
        })
    ent_meta.sort(key=lambda e: (-e["prevalenceInRole"], e["entId"]))

    # Application breakdown
    app_counts: dict[str, int] = {}
    for em in ent_meta:
        app_counts[em["appName"]] = app_counts.get(em["appName"], 0) + 1
    total_ents = len(ent_meta)
    applications = [
        {"appName": app, "appId": "", "entitlementCount": cnt,
         "pctOfRole": round(cnt / total_ents, 4)}
        for app, cnt in app_counts.items()
    ]

    # Role name
    distinct_apps = sorted(app_counts.keys())
    if len(distinct_apps) <= 3:
        app_label = ", ".join(distinct_apps)
    else:
        app_label = ", ".join(distinct_apps[:3]) + f" +{len(distinct_apps) - 3}"
    name = f"{app_label} Birthright (n={member_count})"

    role_id = str(uuid.uuid4())
    return {
        "id": role_id,
        "status": "candidate",
        "roleType": "layer1_universal",
        "sessionId": session_id,
        "name": name,
        "memberCount": member_count,
        "entitlementCount": len(ent_ids),
        "entitlements": " ".join(e["entId"] for e in ent_meta),
        "applications": applications,
        "entitlementMetadata": ent_meta,
        "justificationMetadata": {
            "sessionId": session_id,
            "filterDescription": population_summary.get("filterDescription", ""),
            "filterCriteria": parameters["filterCriteria"],
            "layer": "layer1",
            "populationSize": population_size,
            "thresholds": _thresholds(parameters),
        },
        "confidence": None,
        "memberIds": [population_ids[i] for i in sorted(intersection)],
    }


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
