# pipeline-service/tests/test_api.py
import copy
import uuid

import pytest
from fastapi.testclient import TestClient

import config
import main
from es import client as es_client
from tests.conftest import DEFAULT_TEST_PARAMS, now_iso

pytestmark = pytest.mark.integration


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


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


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_run_success(client, running_session):
    response = client.post("/run", json={"sessionId": running_session})
    assert response.status_code == 200
    assert response.json()["status"] == "complete"


def test_run_failure(client, seed_es):
    session_id = _seed_session(_params(filterCriteria={"department": "Nonexistent"}))
    response = client.post("/run", json={"sessionId": session_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert isinstance(body["error"], str)


def test_run_cancellation(client, seed_es):
    session_id = _seed_session(_params())
    es_client.update_doc(config.INDEX_SESSIONS, session_id,
                         {"cancelRequested": True})
    es_client._client.indices.refresh(index=config.INDEX_SESSIONS)
    response = client.post("/run", json={"sessionId": session_id})
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"


def test_run_missing_session(client, seed_es):
    response = client.post("/run", json={"sessionId": "nonexistent-uuid"})
    assert response.status_code == 200
    assert response.json()["status"] == "failed"


def test_run_missing_field(client):
    response = client.post("/run", json={})
    assert response.status_code == 422


def test_run_invalid_json(client):
    response = client.post(
        "/run", content="not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
