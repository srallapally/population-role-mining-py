# pipeline-service/tests/test_enrichment.py
import pytest

import config
from es import client as es_client
from es.enrichment import fetch_entitlement_metadata

pytestmark = pytest.mark.integration

_META_FIELDS = ("displayName", "appId", "appName", "entitlementType", "criticality")


def _refresh_entitlements():
    es_client._client.indices.refresh(index=config.INDEX_ENTITLEMENTS)


def test_basic_lookup(seed_es):
    result = fetch_entitlement_metadata(
        ["ent-ad-domain-users", "ent-vpn-allusers", "ent-epic-amb-orders"]
    )
    assert len(result) == 3
    for value in result.values():
        for field in _META_FIELDS:
            assert field in value


def test_field_values(seed_es):
    result = fetch_entitlement_metadata(["ent-ad-domain-users"])
    meta = result["ent-ad-domain-users"]
    assert meta["displayName"] == "TGH Domain Users"
    assert meta["appName"] == "Active Directory"
    assert meta["entitlementType"] == "Group"
    assert meta["criticality"] == "Low"


def test_missing_entitlement(seed_es):
    result = fetch_entitlement_metadata(["ent-ad-domain-users", "nonexistent"])
    assert len(result) == 1
    assert "ent-ad-domain-users" in result
    assert "nonexistent" not in result


def test_empty_input(seed_es):
    assert fetch_entitlement_metadata([]) == {}


def test_batching(seed_es):
    fake_ids = [f"ent-fake-{i}" for i in range(1500)]
    for fake_id in fake_ids:
        es_client.index_doc(config.INDEX_ENTITLEMENTS, fake_id, {
            "entitlementId": fake_id,
            "displayName": fake_id,
            "appId": "app-fake",
            "appName": "Fake App",
            "entitlementType": "Group",
            "criticality": "Low",
        })
    _refresh_entitlements()
    try:
        # 1500 IDs exceed the 1000-per-chunk batch size.
        result = fetch_entitlement_metadata(fake_ids)
        assert len(result) == 1500
    finally:
        for fake_id in fake_ids:
            es_client.delete_doc(config.INDEX_ENTITLEMENTS, fake_id)
        _refresh_entitlements()
