# pipeline-service/tests/test_orchestrator.py
import copy
import uuid

import pytest

import config
from es import client as es_client
from pipeline.orchestrator import run_pipeline, PipelineCancelled
from tests.conftest import DEFAULT_TEST_PARAMS, now_iso

pytestmark = pytest.mark.integration


def _params(**overrides):
    params = copy.deepcopy(DEFAULT_TEST_PARAMS)
    params.update(overrides)
    return params


def _seed_session(params, status="running"):
    session_id = str(uuid.uuid4())
    es_client.index_doc(config.INDEX_SESSIONS, session_id, {
        "id": session_id,
        "status": status,
        "sessionOwner": "test-analyst",
        "createdAt": now_iso(),
        "updatedAt": now_iso(),
        "parameters": params,
    })
    es_client._client.indices.refresh(index=config.INDEX_SESSIONS)
    return session_id


def _get_session(session_id):
    return es_client.get(config.INDEX_SESSIONS, session_id)


def _roles_for(session_id):
    es_client._client.indices.refresh(index=config.INDEX_ROLES)
    return es_client.search(
        config.INDEX_ROLES, {"term": {"sessionId": session_id}}, size=100
    )


def test_happy_path_job_roles(running_session):
    run_pipeline(running_session)

    session = _get_session(running_session)
    assert session["status"] == "complete"
    assert session["populationSize"] == 20
    assert session["layer1RoleCount"] >= 1
    assert session["candidateRoleCount"] >= 0
    assert session["completedAt"] is not None

    roles = _roles_for(running_session)
    assert len(roles) == session["layer1RoleCount"] + session["candidateRoleCount"]


def test_happy_path_birthright(seed_es):
    session_id = _seed_session(_params(roleType="birthright"))
    run_pipeline(session_id)

    session = _get_session(session_id)
    assert session["status"] == "complete"
    assert session["layer1RoleCount"] >= 1
    assert not session.get("candidateRoleIds")

    roles = _roles_for(session_id)
    assert all(r["roleType"] == "layer1_universal" for r in roles)


def test_zero_layer2(seed_es):
    session_id = _seed_session(_params(similarityThreshold=0.99))
    run_pipeline(session_id)

    session = _get_session(session_id)
    assert session["status"] == "complete"
    assert session["candidateRoleCount"] == 0
    assert session["layer1RoleCount"] >= 1


def test_layer1_always_present(seed_es):
    # A true zero-layer1 case is not reachable with the seed data:
    # ent-ad-domain-users is held by all 20 users, so at least one
    # universal entitlement always exists. This verifies the code path
    # by checking layer1RoleCount is populated.
    session_id = _seed_session(_params())
    run_pipeline(session_id)

    session = _get_session(session_id)
    assert session["layer1RoleCount"] >= 1


def test_population_zero(seed_es):
    session_id = _seed_session(_params(filterCriteria={"department": "Nonexistent"}))

    with pytest.raises(Exception):
        run_pipeline(session_id)

    session = _get_session(session_id)
    assert session["status"] == "failed"
    assert "matched 0 users" in session["errorDetail"]


def test_population_exceeds_cap(seed_es):
    session_id = _seed_session(_params(maxPopulation=5))

    with pytest.raises(Exception):
        run_pipeline(session_id)

    session = _get_session(session_id)
    assert session["status"] == "failed"
    assert "exceeding cap" in session["errorDetail"]


def test_session_not_found():
    with pytest.raises(ValueError):
        run_pipeline("nonexistent-uuid")


def test_session_wrong_status(seed_es):
    session_id = _seed_session(_params(), status="pending")

    with pytest.raises(ValueError):
        run_pipeline(session_id)

    # The session is not overwritten.
    assert _get_session(session_id)["status"] == "pending"


def test_missing_parameter(seed_es):
    params = _params()
    del params["coverageThreshold"]
    session_id = _seed_session(params)

    with pytest.raises(ValueError, match="Missing required parameter"):
        run_pipeline(session_id)

    # _validate_session raises before the try-block, so the session is
    # NOT written to "failed" — it stays "running".
    session = _get_session(session_id)
    assert session["status"] != "failed"


def test_invalid_role_type(seed_es):
    session_id = _seed_session(_params(roleType="invalid"))

    with pytest.raises(ValueError, match="Invalid roleType"):
        run_pipeline(session_id)

    # Validation error raised before the try-block — not written failed.
    assert _get_session(session_id)["status"] != "failed"


def test_max_population_ceiling(seed_es):
    session_id = _seed_session(_params(maxPopulation=999999))

    with pytest.raises(ValueError, match="exceeds ceiling"):
        run_pipeline(session_id)

    # Validation error raised before the try-block — not written failed.
    assert _get_session(session_id)["status"] != "failed"


def test_failure_writes_error_detail(seed_es):
    session_id = _seed_session(_params(filterCriteria={"department": "Nonexistent"}))

    with pytest.raises(Exception):
        run_pipeline(session_id)

    error_detail = _get_session(session_id)["errorDetail"]
    assert isinstance(error_detail, str)
    assert len(error_detail) > 0


def test_cancellation(seed_es):
    session_id = _seed_session(_params())
    es_client.update_doc(config.INDEX_SESSIONS, session_id,
                         {"cancelRequested": True})
    es_client._client.indices.refresh(index=config.INDEX_SESSIONS)

    with pytest.raises(PipelineCancelled):
        run_pipeline(session_id)

    assert _get_session(session_id)["status"] == "cancelled"


def test_cancellation_cleans_up_roles(seed_es):
    session_id = _seed_session(_params())
    es_client.update_doc(config.INDEX_SESSIONS, session_id,
                         {"cancelRequested": True})
    es_client._client.indices.refresh(index=config.INDEX_SESSIONS)

    with pytest.raises(PipelineCancelled):
        run_pipeline(session_id)

    # Cancellation is caught at the first checkpoint, before any role
    # is written.
    assert _roles_for(session_id) == []


def test_determinism(seed_es):
    def _run_and_collect():
        session_id = _seed_session(_params())
        run_pipeline(session_id)
        return {
            (r["roleType"], r["name"]): (
                r["entitlements"], r["memberCount"], r["confidence"],
            )
            for r in _roles_for(session_id)
        }

    assert _run_and_collect() == _run_and_collect()


def test_graph_metrics_present(seed_es):
    session_id = _seed_session(_params())
    run_pipeline(session_id)

    metrics = _get_session(session_id)["graphMetrics"]
    for key in ("edgeCount", "graphDensity", "singletonCount",
                "largestConnectedComponentPct"):
        assert key in metrics
        assert isinstance(metrics[key], (int, float))


def test_high_residual_prevalence(seed_es):
    session_id = _seed_session(_params())
    run_pipeline(session_id)

    session = _get_session(session_id)
    assert isinstance(session["highResidualPrevalenceEntitlements"], list)
