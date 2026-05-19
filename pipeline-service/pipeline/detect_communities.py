# backend/pipeline/detect_communities.py
import igraph as ig
import leidenalg
import numpy as np
from scipy.sparse import csr_matrix

LEIDEN_SEED = 42


def detect_communities(
    adjacency: csr_matrix,
    population_ids: list[str],
    min_group_size: int,
    max_roles: int,
) -> tuple[list[dict], list[dict], float]:
    """
    Run Leiden on thresholded adjacency matrix. Apply size + count gating.

    Returns:
      communities: [{"communityId": int, "userIds": [str]}]
      suppressed: [{"communityId": int, "size": int, "reason": str}]
      modularity: float
    """
    n = len(population_ids)

    # Build igraph from adjacency — edges in deterministic order (sorted by node pair)
    cx = adjacency.tocoo()
    edges = sorted(
        {(min(i, j), max(i, j)) for i, j in zip(cx.row, cx.col) if i != j}
    )

    g = ig.Graph(n=n, edges=edges)
    partition = leidenalg.find_partition(
        g, leidenalg.ModularityVertexPartition, seed=LEIDEN_SEED
    )
    modularity = partition.modularity

    raw_communities = []
    for community_id, member_indices in enumerate(partition):
        user_ids = [population_ids[i] for i in member_indices]
        raw_communities.append({
            "communityId": community_id,
            "userIds": user_ids,
            "size": len(user_ids),
        })

    # Size gate
    surviving = []
    suppressed = []
    for c in raw_communities:
        if c["size"] < min_group_size:
            suppressed.append({
                "communityId": c["communityId"],
                "size": c["size"],
                "reason": f"below minimum group size of {min_group_size}",
            })
        else:
            surviving.append(c)

    # Count gate — keep top N by size
    if len(surviving) > max_roles:
        surviving_sorted = sorted(surviving, key=lambda c: -c["size"])
        for c in surviving_sorted[max_roles:]:
            suppressed.append({
                "communityId": c["communityId"],
                "size": c["size"],
                "reason": "exceeded maxRoles cap",
            })
        surviving = surviving_sorted[:max_roles]

    communities = [{"communityId": c["communityId"], "userIds": c["userIds"]} for c in surviving]
    return communities, suppressed, float(modularity)
