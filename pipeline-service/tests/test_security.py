# pipeline-service/tests/test_security.py
import copy
import threading
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


def test_api_key_required(client, monkeypatch):
    monkeypatch.setattr(main, "_PIPELINE_KEY", "test-secret")
    response = client.post("/run", json={"sessionId": "any"})
    assert response.status_code == 401


def test_api_key_wrong(client, monkeypatch):
    monkeypatch.setattr(main, "_PIPELINE_KEY", "test-secret")
    response = client.post(
        "/run", json={"sessionId": "any"},
        headers={"X-Pipeline-Key": "wrong"},
    )
    assert response.status_code == 401


def test_api_key_correct(client, monkeypatch, seed_es):
    monkeypatch.setattr(main, "_PIPELINE_KEY", "test-secret")
    session_id = _seed_session(_params())
    response = client.post(
        "/run", json={"sessionId": session_id},
        headers={"X-Pipeline-Key": "test-secret"},
    )
    assert response.status_code == 200


def test_api_key_disabled(client, monkeypatch):
    monkeypatch.setattr(main, "_PIPELINE_KEY", "")
    response = client.post("/run", json={"sessionId": "any"})
    assert response.status_code != 401


def test_health_bypasses_key(client, monkeypatch):
    monkeypatch.setattr(main, "_PIPELINE_KEY", "test-secret")
    response = client.get("/health")
    assert response.status_code == 200


def test_concurrency_guard(client, monkeypatch, seed_es):
    # Force a capacity of 1 and make the running pipeline block until
    # the test releases it.
    entered = threading.Event()
    release = threading.Event()

    def blocking_run(session_id):
        entered.set()
        release.wait(timeout=10)

    monkeypatch.setattr(main, "run_pipeline", blocking_run)
    monkeypatch.setattr(main, "_running", threading.Semaphore(1))
    monkeypatch.setattr(main, "_MAX_CONCURRENT_RUNS", 1)

    first_id = _seed_session(_params())
    second_id = _seed_session(_params())

    def first_request():
        with TestClient(main.app) as first_client:
            first_client.post("/run", json={"sessionId": first_id})

    thread = threading.Thread(target=first_request)
    thread.start()
    try:
        # Once blocking_run is entered, the single permit is held.
        assert entered.wait(timeout=5)
        response = client.post("/run", json={"sessionId": second_id})
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "failed"
        assert "capacity" in body["error"]
    finally:
        release.set()
        thread.join(timeout=10)


def test_filter_key_injection(client, seed_es):
    session_id = _seed_session(_params(filterCriteria={"<script>": "value"}))
    response = client.post("/run", json={"sessionId": session_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "Invalid filter key" in body["error"]


def test_session_validation_missing_params(client, seed_es):
    # An empty parameters dict hits _validate_session's "Session has no
    # parameters" branch; a dict missing a required field exercises the
    # "Missing required parameter" branch this test targets.
    params = _params()
    del params["coverageThreshold"]
    session_id = _seed_session(params)
    response = client.post("/run", json={"sessionId": session_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "Missing required" in body["error"]


def test_session_validation_ceiling(client, seed_es):
    session_id = _seed_session(_params(maxPopulation=50000))
    response = client.post("/run", json={"sessionId": session_id})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert "exceeds ceiling" in body["error"]
