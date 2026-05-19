# Pipeline Service

A Python microservice that runs role-mining algorithms against
Elasticsearch data. It is one half of a two-service architecture:

- A **Node/Express API** handles analyst-facing HTTP, session
  lifecycle, and role editing.
- This **pipeline service** handles computation: it reads identity and
  entitlement data from Elasticsearch, mines roles, and writes the
  results back to Elasticsearch.

The two services share state only through Elasticsearch indexes — the
Node API creates a session document, asks this service to run, and
this service writes roles and the session's terminal state back.

## How a run works

The Node API creates a session document (`status = "running"`) in the
sessions index, then calls `POST /run` with the session ID. This
service loads the session, executes the six-step pipeline, and writes
the outcome back to Elasticsearch. The HTTP response reports the
outcome but is informational only — the session and role documents in
Elasticsearch are the source of truth.

```
Node API ──create session──▶ Elasticsearch
Node API ──POST /run────────▶ Pipeline Service ──read identities/entitlements──▶ Elasticsearch
                                              ──write roles + session result──▶ Elasticsearch
```

## The pipeline

`pipeline/orchestrator.py` drives six steps. A run is either
`birthright` (Layer 1 only) or `job_roles` (the full two-layer
pipeline), selected by the session's `roleType` parameter.

| Step | Module | Purpose |
|---|---|---|
| 1 | `filter_population.py` | Resolve the population from the identity index using the analyst's filter criteria plus always-on system filters (active status, human account type). Returns user IDs, a summary, and each user's entitlement assignments. |
| 2 | `build_sparse_matrix.py` | Build a binary user × entitlement CSR matrix. Entitlements held by fewer than `noiseFilterValue` users are dropped. |
| 3 | `compute_universal.py` | Prevalence scan: entitlements above `universalThreshold` are "universal". Cluster them by holder-set co-occurrence into **Layer 1 birthright roles**. Produce the residual matrix (universal columns zeroed). |
| 4 | `build_similarity_graph.py` | Build a user-similarity graph from the residual matrix using Jaccard similarity, keeping edges at or above `similarityThreshold`. |
| 5 | `detect_communities.py` | Run Leiden community detection on the graph. Apply the `minGroupSize` and `maxRoles` gates. |
| 6 | `compose_roles.py` | For each surviving community, compute a **Layer 2 candidate role**: two-tier entitlement tiering, cohesion scoring, application breakdown, and outlier analysis. |

For a `birthright` run the orchestrator stops after Step 3 and writes
only the Layer 1 roles. For a `job_roles` run it runs all six steps
and writes both layers.

Entitlement display metadata (`displayName`, `appName`, etc.) is
fetched once from the entitlement catalog via
`es/enrichment.fetch_entitlement_metadata()` and passed into Steps 3
and 6.

### Cancellation

Between every step the orchestrator re-reads the session document and
checks `cancelRequested`. If the Node API has set it, the orchestrator
marks the session `cancelled`, deletes any roles already written for
the session, and raises `PipelineCancelled`.

### Failure

Any other exception during a step marks the session `failed` with an
`errorDetail` message, then re-raises. Validation errors detected
before the pipeline starts (missing/invalid session) raise directly.

## Elasticsearch indexes

Index names and field names are environment-overridable, so the same
code runs against mock indexes in development and production indexes
in deployment — only the configuration changes, not the pipeline
logic.

- **Identities** (`INDEX_IDENTITIES`) — one document per user, with a
  nested `assignments` array. Read in Step 1.
- **Entitlements** (`INDEX_ENTITLEMENTS`) — the entitlement catalog.
  Read for display-metadata enrichment.
- **Sessions** (`INDEX_SESSIONS`) — one mutable document per run.
  Created by the Node API; updated by this service at completion,
  failure, or cancellation.
- **Roles** (`INDEX_ROLES`) — one document per mined role. Written by
  this service.

## HTTP API

`main.py` exposes a small FastAPI app. It is internal — only the Node
API is expected to call it.

- `POST /run` — body `{"sessionId": "..."}`. Runs the pipeline for
  that session. Always returns **HTTP 200**; the `status` field
  distinguishes the outcome:
  - `{"status": "complete", "sessionId": "..."}`
  - `{"status": "cancelled", "sessionId": "..."}`
  - `{"status": "failed", "sessionId": "...", "error": "..."}`

  A pipeline failure is a business outcome (bad filter, empty
  population, ES error), not a server error, so it is not a 500 —
  the Node API always receives a parseable JSON response.

- `GET /health` — returns `{"status": "ok"}`.

### Protections

- **Authentication** — if `PIPELINE_API_KEY` is set, every request
  except `GET /health` must carry a matching `X-Pipeline-Key` header,
  or it is rejected with 401. If the variable is unset, the check is a
  no-op (frictionless local development).
- **Concurrency guard** — at most `MAX_CONCURRENT_RUNS` pipelines run
  at once; further requests get a capacity `failed` response instead
  of overloading the process.
- **Execution timeout** — a run exceeding `RUN_TIMEOUT_SECONDS` is
  reported as a timeout failure and the session is marked `failed`.
- **Input validation** — the orchestrator re-validates the session
  (status, required parameters, population ceiling, role type) before
  computing, and filter-criteria keys are validated against a strict
  pattern to guard against query injection.

## Configuration

All configuration is read from environment variables in `config.py`,
with the defaults below.

### Elasticsearch connection
| Variable | Default |
|---|---|
| `ES_HOST` | `http://localhost:9200` |
| `ES_API_KEY` | _(none)_ |
| `ES_SCROLL_SIZE` | `1000` |
| `ES_SCROLL_TIMEOUT` | `2m` |

### Index names
| Variable | Default |
|---|---|
| `INDEX_IDENTITIES` | `mock_identities` |
| `INDEX_ENTITLEMENTS` | `mock_entitlements` |
| `INDEX_SESSIONS` | `mock_sessions` |
| `INDEX_ROLES` | `mock_roles` |

### Identity field mapping
`FIELD_USER_ID` (`userId`), `FIELD_ACCOUNT_STATUS` (`accountStatus`),
`FIELD_ACCOUNT_TYPE` (`accountType`), `FIELD_ASSIGNMENTS`
(`assignments`), `FIELD_ASSIGNMENT_ENT_ID` (`entitlementId`).

System filter values: `SYSTEM_FILTER_STATUS_VALUE` (`active`),
`SYSTEM_FILTER_TYPE_VALUE` (`human`).

### Entitlement field mapping
`FIELD_ENT_ID` (`entitlementId`), `FIELD_ENT_DISPLAY_NAME`
(`displayName`), `FIELD_ENT_APP_ID` (`appId`), `FIELD_ENT_APP_NAME`
(`appName`), `FIELD_ENT_TYPE` (`entitlementType`),
`FIELD_ENT_CRITICALITY` (`criticality`).

### Server / runtime
| Variable | Default |
|---|---|
| `PIPELINE_PORT` | `8001` |
| `PIPELINE_API_KEY` | _(empty — auth disabled)_ |
| `MAX_CONCURRENT_RUNS` | `5` |
| `RUN_TIMEOUT_SECONDS` | `300` |

Pipeline thresholds (`universalThreshold`, `coverageThreshold`,
`softThreshold`, `similarityThreshold`, `minGroupSize`, `maxRoles`,
`noiseFilterValue`, etc.) are **not** environment variables — they are
per-session parameters stored on the session document by the Node API.

## Running the service

```sh
cd pipeline-service
pip install -r requirements.txt
python main.py            # serves on PIPELINE_PORT (default 8001)
```

`main.py` starts Uvicorn. On startup it probes Elasticsearch
reachability and logs the result, but does not block startup if ES is
down — the first `/run` call will fail with a clear error instead.

## Project layout

```
pipeline-service/
  config.py              flat env-driven configuration
  main.py                FastAPI app: /run and /health
  es/
    client.py            thin Elasticsearch wrapper
    enrichment.py        entitlement metadata lookup
  pipeline/
    orchestrator.py      run_pipeline — drives the six steps
    filter_population.py        Step 1
    build_sparse_matrix.py      Step 2
    compute_universal.py        Step 3
    build_similarity_graph.py   Step 4
    detect_communities.py       Step 5
    compose_roles.py            Step 6
  tests/                 pytest suite (see below)
  requirements.txt
```

## Testing

Tests live in `tests/`. They fall into two tiers:

- **Unit tests** — pure computation on in-memory fixtures
  (`test_build_sparse_matrix.py`, `test_compute_universal.py`,
  `test_build_similarity_graph.py`, `test_detect_communities.py`,
  `test_compose_roles.py`).
- **Integration tests** — marked `@pytest.mark.integration`; they
  require a running Elasticsearch and exercise the ES client,
  enrichment, `filter_population`, the orchestrator, and the HTTP API
  end to end against seeded mock indexes.

`tests/conftest.py` provides the shared fixtures: seed identity and
entitlement data, index creation/teardown, and a seeded running
session.

```sh
pip install pytest httpx          # test-only dependencies
pytest                            # run the full suite
pytest -m "not integration"       # unit tests only
pytest -m integration             # integration tests (needs ES)
```
