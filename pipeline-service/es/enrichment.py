# pipeline-service/es/enrichment.py
import config
from es import client as es_client

_CHUNK_SIZE = 1000


def fetch_entitlement_metadata(ent_ids: list[str]) -> dict[str, dict]:
    """Fetch display metadata for entitlements from the entitlement catalog.

    Returns a lookup keyed by entitlement ID. IDs absent from the catalog
    are silently omitted; callers fall back to .get() defaults.
    """
    if not ent_ids:
        return {}

    lookup: dict[str, dict] = {}
    for start in range(0, len(ent_ids), _CHUNK_SIZE):
        chunk = ent_ids[start:start + _CHUNK_SIZE]
        docs = es_client.search(
            config.INDEX_ENTITLEMENTS,
            {"bool": {"filter": {"terms": {config.FIELD_ENT_ID: chunk}}}},
            size=len(chunk),
        )
        for doc in docs:
            lookup[doc[config.FIELD_ENT_ID]] = {
                "displayName": doc.get(config.FIELD_ENT_DISPLAY_NAME, ""),
                "appId": doc.get(config.FIELD_ENT_APP_ID, ""),
                "appName": doc.get(config.FIELD_ENT_APP_NAME, ""),
                "entitlementType": doc.get(config.FIELD_ENT_TYPE, ""),
                "criticality": doc.get(config.FIELD_ENT_CRITICALITY, ""),
            }

    return lookup
