# pipeline-service/tests/test_filter_population.py
import pytest

import config
from es import client as es_client
from pipeline.filter_population import filter_population

pytestmark = pytest.mark.integration


def _refresh_identities():
    es_client._client.indices.refresh(index=config.INDEX_IDENTITIES)


def _index_extra(doc_id, doc):
    es_client.index_doc(config.INDEX_IDENTITIES, doc_id, doc)
    _refresh_identities()


def _delete_extra(doc_id):
    es_client.delete_doc(config.INDEX_IDENTITIES, doc_id)
    _refresh_identities()


def test_basic_filter(seed_es):
    population_ids, _, _ = filter_population(
        {"department": "Ambulatory Informatics"}, 10000
    )
    assert len(population_ids) == 20


def test_multi_value_filter(seed_es):
    population_ids, _, _ = filter_population(
        {"jobCode": ["EPIC_LINK", "NONEXISTENT"]}, 10000
    )
    assert len(population_ids) == 20


def test_combined_filter(seed_es):
    population_ids, _, _ = filter_population(
        {"department": "Ambulatory Informatics", "jobCode": "EPIC_LINK"}, 10000
    )
    assert len(population_ids) == 20


def test_system_filters_applied(seed_es):
    _index_extra("usr-extra-inactive", {
        "userId": "usr-extra-inactive",
        "userName": "usr-extra-inactive",
        "accountStatus": "inactive",
        "accountType": "human",
        "department": "Ambulatory Informatics",
        "jobCode": "EPIC_LINK",
        "assignments": [],
    })
    try:
        population_ids, _, _ = filter_population(
            {"department": "Ambulatory Informatics"}, 10000
        )
        # The inactive account is excluded by the system filter.
        assert len(population_ids) == 20
    finally:
        _delete_extra("usr-extra-inactive")


def test_system_filter_account_type(seed_es):
    _index_extra("usr-extra-service", {
        "userId": "usr-extra-service",
        "userName": "usr-extra-service",
        "accountStatus": "active",
        "accountType": "service",
        "department": "Ambulatory Informatics",
        "jobCode": "EPIC_LINK",
        "assignments": [],
    })
    try:
        population_ids, _, _ = filter_population(
            {"department": "Ambulatory Informatics"}, 10000
        )
        # The service account is excluded by the system filter.
        assert len(population_ids) == 20
    finally:
        _delete_extra("usr-extra-service")


def test_zero_matches(seed_es):
    with pytest.raises(ValueError, match="matched 0 users"):
        filter_population({"department": "Nonexistent"}, 10000)


def test_exceeds_cap(seed_es):
    with pytest.raises(ValueError, match="exceeding cap of 5"):
        filter_population({"department": "Ambulatory Informatics"}, 5)


def test_at_cap(seed_es):
    population_ids, _, _ = filter_population(
        {"department": "Ambulatory Informatics"}, 20
    )
    assert len(population_ids) == 20


def test_population_ids_sorted(seed_es):
    population_ids, _, _ = filter_population(
        {"department": "Ambulatory Informatics"}, 10000
    )
    assert population_ids == sorted(population_ids)


def test_assignments_extracted(seed_es):
    _, _, population_assignments = filter_population(
        {"department": "Ambulatory Informatics"}, 10000
    )
    assert len(population_assignments) == 20
    for ent_ids in population_assignments.values():
        assert isinstance(ent_ids, list)
        assert len(ent_ids) > 0
        assert all(isinstance(e, str) for e in ent_ids)


def test_assignments_match_seed(seed_es):
    _, _, population_assignments = filter_population(
        {"department": "Ambulatory Informatics"}, 10000
    )
    expected = {
        "ent-ad-domain-users", "ent-vpn-allusers", "ent-mfa-allusers",
        "ent-epic-amb-orders", "ent-epic-amb-notes",
        "ent-epic-amb-schedule", "ent-epic-amb-results",
    }
    assert expected <= set(population_assignments["usr-001"])


def test_summary_format(seed_es):
    criteria = {"department": "Ambulatory Informatics"}
    _, population_summary, _ = filter_population(criteria, 10000)
    assert population_summary["count"] == 20
    assert "department" in population_summary["filterDescription"]
    assert population_summary["filterCriteria"] == criteria


def test_invalid_filter_key(seed_es):
    with pytest.raises(ValueError, match="Invalid filter key"):
        filter_population({"<script>": "value"}, 10000)
