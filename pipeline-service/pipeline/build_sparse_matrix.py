# pipeline-service/pipeline/build_sparse_matrix.py
import numpy as np
from scipy.sparse import csr_matrix


def build_sparse_matrix(
    population_ids: list[str],
    population_assignments: dict[str, list[str]],
    noise_filter_value: int,
) -> tuple[csr_matrix, dict[str, int], int]:
    """Build a binary CSR matrix (users x entitlements) from in-memory assignments.

    Rows follow population_ids order (already sorted lexicographically by
    filter_population). Columns are the entitlements that survive the noise
    filter, sorted lexicographically. Entitlements held by fewer than
    noise_filter_value users are dropped.

    Returns: (matrix, entitlement_index, dropped_count)
    """
    holder_counts: dict[str, int] = {}
    for ents in population_assignments.values():
        for ent_id in set(ents):
            holder_counts[ent_id] = holder_counts.get(ent_id, 0) + 1

    surviving = sorted(
        ent_id for ent_id, count in holder_counts.items()
        if count >= noise_filter_value
    )
    dropped_count = len(holder_counts) - len(surviving)

    entitlement_index = {ent_id: idx for idx, ent_id in enumerate(surviving)}
    user_index = {user_id: idx for idx, user_id in enumerate(population_ids)}

    rows: list[int] = []
    cols: list[int] = []
    for user_id, ents in population_assignments.items():
        row = user_index[user_id]
        for ent_id in ents:
            col = entitlement_index.get(ent_id)
            if col is not None:
                rows.append(row)
                cols.append(col)

    data = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix(
        (data, (rows, cols)),
        shape=(len(population_ids), len(entitlement_index)),
    )
    matrix.data[:] = 1.0

    return matrix, entitlement_index, dropped_count
