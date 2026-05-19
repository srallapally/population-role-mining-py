# pipeline-service/tests/conftest.py
import uuid
from datetime import datetime, timezone

import pytest

import config
from es import client as es_client


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def pytest_configure(config):  # `config` here is the pytest Config, not the project module
    config.addinivalue_line(
        "markers", "integration: requires running ES instance"
    )


# --- Resolved parameters --------------------------------------------------

# minGroupSize is 3 (not the config.py default of 30) because the test
# population is only 20 users. Everything else mirrors config.py defaults.
DEFAULT_TEST_PARAMS = {
    "filterCriteria": {"department": "Ambulatory Informatics"},
    "roleType": "job_roles",
    "universalThreshold": 0.90,
    "coverageThreshold": 0.80,
    "softThreshold": 0.50,
    "similarityThreshold": 0.30,
    "minGroupSize": 3,
    "maxRoles": 25,
    "outlierThreshold": 0.30,
    "birthrightCooccurrenceThreshold": 0.95,
    "maxPopulation": 10000,
    "noiseFilterValue": 2,
    "noiseFilterFormula": "max(2, floor(3 * 0.80) - 1)",
}


# --- Seed: entitlement catalog -------------------------------------------

def _ent(eid, display, app_id, app_name, etype, criticality):
    return {
        "entitlementId": eid,
        "displayName": display,
        "appId": app_id,
        "appName": app_name,
        "entitlementType": etype,
        "criticality": criticality,
        "description": f"{display} entitlement",
    }


SEED_ENTITLEMENTS = [
    # Universal
    _ent("ent-ad-domain-users", "TGH Domain Users", "app-ad", "Active Directory", "Group", "Low"),
    _ent("ent-vpn-allusers", "VPN All Users", "app-vpn", "Pulse Secure VPN", "Group", "Low"),
    _ent("ent-mfa-allusers", "MFA All Users", "app-mfa", "Ping MFA", "Policy", "Low"),
    # Cluster A — Epic
    _ent("ent-epic-amb-orders", "Epic Ambulatory Orders", "app-epic", "Epic", "Role", "High"),
    _ent("ent-epic-amb-notes", "Epic Ambulatory Notes", "app-epic", "Epic", "Role", "Medium"),
    _ent("ent-epic-amb-schedule", "Epic Ambulatory Schedule", "app-epic", "Epic", "Role", "Medium"),
    _ent("ent-epic-amb-results", "Epic Ambulatory Results", "app-epic", "Epic", "Role", "Medium"),
    # Cluster B — ServiceNow
    _ent("ent-snow-incident", "ServiceNow Incident", "app-snow", "ServiceNow", "Role", "Medium"),
    _ent("ent-snow-change", "ServiceNow Change", "app-snow", "ServiceNow", "Role", "Medium"),
    _ent("ent-snow-cmdb", "ServiceNow CMDB", "app-snow", "ServiceNow", "Role", "Medium"),
    _ent("ent-snow-reports", "ServiceNow Reports", "app-snow", "ServiceNow", "Role", "Low"),
    # Extras — for outlier testing
    _ent("ent-extra-a1", "Misc A One", "app-misc-a", "Misc App A", "Group", "Low"),
    _ent("ent-extra-a2", "Misc A Two", "app-misc-a", "Misc App A", "Group", "Low"),
    _ent("ent-extra-a3", "Misc A Three", "app-misc-a", "Misc App A", "Group", "Low"),
    _ent("ent-extra-b1", "Misc B One", "app-misc-b", "Misc App B", "Group", "Low"),
    _ent("ent-extra-b2", "Misc B Two", "app-misc-b", "Misc App B", "Group", "Low"),
    # Unique — for singleton testing (each its own app)
    _ent("ent-unique-16", "Unique 16", "app-uniq-16", "Unique App 16", "Group", "Low"),
    _ent("ent-unique-17", "Unique 17", "app-uniq-17", "Unique App 17", "Group", "Low"),
    _ent("ent-unique-18", "Unique 18", "app-uniq-18", "Unique App 18", "Group", "Low"),
    _ent("ent-unique-19", "Unique 19", "app-uniq-19", "Unique App 19", "Group", "Low"),
    _ent("ent-unique-20", "Unique 20", "app-uniq-20", "Unique App 20", "Group", "Low"),
]


# --- Seed: identities -----------------------------------------------------

def _build_seed_identities():
    identities = []
    for n in range(1, 21):
        user_id = f"usr-{n:03d}"
        ents = ["ent-ad-domain-users"]          # all 20 hold AD
        if n != 20:
            ents.append("ent-vpn-allusers")     # all but usr-020 (VPN = 19/20)
        if n != 19:
            ents.append("ent-mfa-allusers")     # all but usr-019 (MFA = 19/20)

        if 1 <= n <= 8:                          # Cluster A — Epic
            ents.append("ent-epic-amb-orders")          # 8/8
            if 1 <= n <= 7:
                ents.append("ent-epic-amb-notes")       # 7/8
                ents.append("ent-epic-amb-schedule")    # 7/8
            if 1 <= n <= 6:
                ents.append("ent-epic-amb-results")     # 6/8
            if n == 3:
                ents += ["ent-extra-a1", "ent-extra-a2", "ent-extra-a3"]
        elif 9 <= n <= 15:                       # Cluster B — ServiceNow
            ents.append("ent-snow-incident")            # 7/7
            if 9 <= n <= 14:
                ents.append("ent-snow-change")          # 6/7
                ents.append("ent-snow-cmdb")            # 6/7
            if 9 <= n <= 13:
                ents.append("ent-snow-reports")         # 5/7
            if n == 12:
                ents += ["ent-extra-b1", "ent-extra-b2"]
        else:                                    # usr-016..020 — singletons
            ents.append(f"ent-unique-{n}")

        identities.append({
            "userId": user_id,
            "userName": user_id,
            "accountStatus": "active",
            "accountType": "human",
            "department": "Ambulatory Informatics",
            "jobCode": "EPIC_LINK",
            "assignments": [
                {"entitlementId": e, "entitlementName": e, "grantedAt": "2024-01-01"}
                for e in ents
            ],
        })
    return identities


SEED_IDENTITIES = _build_seed_identities()


# --- Index mappings (from mock_es_schemas.md) ----------------------------

_IDENTITIES_MAPPING = {
    "properties": {
        "userId": {"type": "keyword"},
        "userName": {"type": "keyword"},
        "accountStatus": {"type": "keyword"},
        "accountType": {"type": "keyword"},
        "department": {"type": "keyword"},
        "jobCode": {"type": "keyword"},
        "jobTitle": {"type": "keyword"},
        "costCenter": {"type": "keyword"},
        "location": {"type": "keyword"},
        "managerId": {"type": "keyword"},
        "employeeType": {"type": "keyword"},
        "assignments": {
            "type": "nested",
            "properties": {
                "entitlementId": {"type": "keyword"},
                "entitlementName": {"type": "keyword"},
                "grantedAt": {"type": "date"},
            },
        },
    }
}

_ENTITLEMENTS_MAPPING = {
    "properties": {
        "entitlementId": {"type": "keyword"},
        "displayName": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "appId": {"type": "keyword"},
        "appName": {"type": "keyword"},
        "entitlementType": {"type": "keyword"},
        "criticality": {"type": "keyword"},
        "description": {"type": "text"},
    }
}

_SESSIONS_MAPPING = {
    "properties": {
        "id": {"type": "keyword"},
        "status": {"type": "keyword"},
        "sessionOwner": {"type": "keyword"},
        "lastUpdatedBy": {"type": "keyword"},
        "createdAt": {"type": "date"},
        "updatedAt": {"type": "date"},
        "completedAt": {"type": "date"},
        "parameters": {
            "type": "object",
            "properties": {
                "filterCriteria": {"type": "object", "enabled": False},
                "filterDescription": {"type": "text"},
                "roleType": {"type": "keyword"},
                "universalThreshold": {"type": "float"},
                "coverageThreshold": {"type": "float"},
                "softThreshold": {"type": "float"},
                "similarityThreshold": {"type": "float"},
                "minGroupSize": {"type": "integer"},
                "maxRoles": {"type": "integer"},
                "outlierThreshold": {"type": "float"},
                "birthrightCooccurrenceThreshold": {"type": "float"},
                "maxPopulation": {"type": "integer"},
                "noiseFilterValue": {"type": "integer"},
                "noiseFilterFormula": {"type": "keyword"},
            },
        },
        "populationSize": {"type": "integer"},
        "totalEntitlementsConsidered": {"type": "integer"},
        "totalEntitlementsDropped": {"type": "integer"},
        "layer1RoleIds": {"type": "keyword"},
        "layer1RoleCount": {"type": "integer"},
        "candidateRoleIds": {"type": "keyword"},
        "candidateRoleCount": {"type": "integer"},
        "singletonCount": {"type": "integer"},
        "graphMetrics": {
            "type": "object",
            "properties": {
                "edgeCount": {"type": "integer"},
                "graphDensity": {"type": "float"},
                "singletonCount": {"type": "integer"},
                "largestConnectedComponentPct": {"type": "float"},
            },
        },
        "highResidualPrevalenceEntitlements": {
            "type": "nested",
            "properties": {
                "entitlementId": {"type": "keyword"},
                "displayName": {"type": "keyword"},
                "appName": {"type": "keyword"},
                "prevalenceInPop": {"type": "float"},
            },
        },
        "populationSummary": {"type": "object", "enabled": False},
        "errorDetail": {"type": "text"},
        "resolvedUniversalThreshold": {"type": "float"},
        "resolvedCoverageThreshold": {"type": "float"},
        "resolvedSoftThreshold": {"type": "float"},
        "resolvedSimilarityThreshold": {"type": "float"},
        "resolvedMinGroupSize": {"type": "integer"},
        "resolvedMaxRoles": {"type": "integer"},
        "resolvedOutlierThreshold": {"type": "float"},
        "resolvedBirthrightCooccurrenceThreshold": {"type": "float"},
        "resolvedNoiseFilterValue": {"type": "integer"},
    }
}

_ROLES_MAPPING = {
    "properties": {
        "id": {"type": "keyword"},
        "status": {"type": "keyword"},
        "roleType": {"type": "keyword"},
        "sessionId": {"type": "keyword"},
        "name": {"type": "text", "fields": {"raw": {"type": "keyword"}}},
        "memberCount": {"type": "integer"},
        "entitlementCount": {"type": "integer"},
        "confidence": {"type": "float"},
        "entitlements": {"type": "keyword"},
        "memberIds": {"type": "keyword"},
        "applications": {
            "type": "nested",
            "properties": {
                "appName": {"type": "keyword"},
                "appId": {"type": "keyword"},
                "entitlementCount": {"type": "integer"},
                "pctOfRole": {"type": "float"},
            },
        },
        "entitlementMetadata": {
            "type": "nested",
            "properties": {
                "entitlementId": {"type": "keyword"},
                "displayName": {"type": "keyword"},
                "appId": {"type": "keyword"},
                "appName": {"type": "keyword"},
                "entitlementType": {"type": "keyword"},
                "criticality": {"type": "keyword"},
                "prevalenceInRole": {"type": "float"},
                "tier": {"type": "keyword"},
            },
        },
        "justificationMetadata": {
            "type": "object",
            "properties": {
                "sessionId": {"type": "keyword"},
                "filterDescription": {"type": "text"},
                "filterCriteria": {"type": "object", "enabled": False},
                "layer": {"type": "keyword"},
                "populationSize": {"type": "integer"},
                "thresholds": {"type": "object", "enabled": False},
                "communityId": {"type": "integer"},
                "communityModularity": {"type": "float"},
                "cohesionInterpretation": {"type": "keyword"},
                "outliers": {
                    "type": "nested",
                    "properties": {
                        "userId": {"type": "keyword"},
                        "underProvisioningScore": {"type": "float"},
                        "overProvisioningScore": {"type": "float"},
                        "underProvisioned": {"type": "keyword"},
                        "overProvisioned": {"type": "keyword"},
                        "plainLanguage": {"type": "text"},
                    },
                },
            },
        },
        "lastScanned": {"type": "date"},
        "analystEdited": {"type": "boolean"},
    }
}

_INDEX_MAPPINGS = {
    config.INDEX_IDENTITIES: _IDENTITIES_MAPPING,
    config.INDEX_ENTITLEMENTS: _ENTITLEMENTS_MAPPING,
    config.INDEX_SESSIONS: _SESSIONS_MAPPING,
    config.INDEX_ROLES: _ROLES_MAPPING,
}

# es/client.py does not expose index admin (create/delete/refresh/
# delete_by_query), so test setup reaches the underlying client directly.
_es = es_client._client


# --- Fixtures -------------------------------------------------------------

@pytest.fixture(scope="session")
def seed_es():
    """Create indexes with mappings, index seed data, refresh.
    Teardown: delete all four indexes."""
    for index, mapping in _INDEX_MAPPINGS.items():
        _es.indices.delete(index=index, ignore_unavailable=True)
        _es.indices.create(index=index, mappings=mapping)

    for identity in SEED_IDENTITIES:
        es_client.index_doc(config.INDEX_IDENTITIES, identity["userId"], identity)
    for entitlement in SEED_ENTITLEMENTS:
        es_client.index_doc(
            config.INDEX_ENTITLEMENTS, entitlement["entitlementId"], entitlement
        )

    for index in _INDEX_MAPPINGS:
        _es.indices.refresh(index=index)

    yield

    for index in _INDEX_MAPPINGS:
        _es.indices.delete(index=index, ignore_unavailable=True)


@pytest.fixture(autouse=True)
def clean_output_indexes(seed_es):
    """Before each test, delete all docs from the output indexes."""
    for index in (config.INDEX_SESSIONS, config.INDEX_ROLES):
        _es.delete_by_query(
            index=index,
            query={"match_all": {}},
            refresh=True,
            conflicts="proceed",
        )
    yield


@pytest.fixture
def running_session(seed_es):
    """Seed a valid running session in ES. Return its session_id."""
    session_id = str(uuid.uuid4())
    session_doc = {
        "id": session_id,
        "status": "running",
        "sessionOwner": "test-analyst",
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "parameters": DEFAULT_TEST_PARAMS,
    }
    es_client.index_doc(config.INDEX_SESSIONS, session_id, session_doc)
    _es.indices.refresh(index=config.INDEX_SESSIONS)
    return session_id


@pytest.fixture
def population_data():
    """Return (population_ids, population_assignments) matching the seed
    data — for Tier 1 tests that do not need ES."""
    population_ids = sorted(i["userId"] for i in SEED_IDENTITIES)
    population_assignments = {
        i["userId"]: [a["entitlementId"] for a in i["assignments"]]
        for i in SEED_IDENTITIES
    }
    return population_ids, population_assignments


@pytest.fixture
def ent_lookup():
    """Return entitlement metadata keyed by entitlement ID, matching the
    seed catalog — for Tier 1 tests that do not need ES."""
    return {
        e["entitlementId"]: {
            "displayName": e["displayName"],
            "appId": e["appId"],
            "appName": e["appName"],
            "entitlementType": e["entitlementType"],
            "criticality": e["criticality"],
        }
        for e in SEED_ENTITLEMENTS
    }
