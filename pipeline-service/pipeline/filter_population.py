# pipeline-service/pipeline/filter_population.py
import re

import config
from es import client as es_client

_FILTER_KEY_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_]{0,63}$')


def filter_population(
    filter_criteria: dict, max_population: int
) -> tuple[list[str], dict, dict[str, list[str]]]:
    """Resolve the population for a role-mining session from Elasticsearch.

    Builds a bool query from the analyst's filter criteria plus the
    always-on system filters (active status, human account type),
    enforces the population cap, then returns the matched user IDs,
    a summary, and each user's entitlement assignments.
    """
    for key in filter_criteria:
        if not _FILTER_KEY_PATTERN.match(key):
            raise ValueError(f"Invalid filter key: '{key}'")

    must: list[dict] = []
    for key, value in filter_criteria.items():
        if isinstance(value, list):
            must.append({"terms": {key: value}})
        else:
            must.append({"term": {key: value}})

    must.append(
        {"term": {config.FIELD_ACCOUNT_STATUS: config.SYSTEM_FILTER_STATUS_VALUE}}
    )
    must.append(
        {"term": {config.FIELD_ACCOUNT_TYPE: config.SYSTEM_FILTER_TYPE_VALUE}}
    )

    query = {"bool": {"must": must}}

    count = es_client.count(config.INDEX_IDENTITIES, query)
    if count == 0:
        raise ValueError(
            f"Population filter matched 0 users. Filter: {filter_criteria}"
        )
    if count > max_population:
        raise ValueError(
            f"Population filter matched {count} users, exceeding cap of "
            f"{max_population}. Narrow the filter criteria."
        )

    docs = es_client.scroll_all(config.INDEX_IDENTITIES, query)

    population_assignments: dict[str, list[str]] = {}
    for doc in docs:
        user_id = doc[config.FIELD_USER_ID]
        ent_ids: list[str] = []
        for entry in doc.get(config.FIELD_ASSIGNMENTS, []):
            ent_id = entry.get(config.FIELD_ASSIGNMENT_ENT_ID)
            if ent_id:
                ent_ids.append(ent_id)
        population_assignments[user_id] = ent_ids

    population_ids = sorted(population_assignments.keys())

    parts: list[str] = []
    for key, value in filter_criteria.items():
        if isinstance(value, list):
            parts.append(f"{key} IN {value!r}")
        else:
            parts.append(f"{key} = {value}")

    population_summary = {
        "count": len(population_ids),
        "filterDescription": " AND ".join(parts),
        "filterCriteria": filter_criteria,
    }

    return population_ids, population_summary, population_assignments
