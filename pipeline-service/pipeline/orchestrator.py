# pipeline-service/pipeline/orchestrator.py
from datetime import datetime, timezone

import config
from es import client as es_client
from es.enrichment import fetch_entitlement_metadata
from pipeline.filter_population import filter_population
from pipeline.build_sparse_matrix import build_sparse_matrix
from pipeline.compute_universal import compute_universal
from pipeline.build_similarity_graph import build_similarity_graph
from pipeline.detect_communities import detect_communities
from pipeline.compose_roles import compose_roles


class PipelineCancelled(Exception):
    pass


_MAX_POPULATION_CEILING = 10000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_cancelled(session_id: str) -> None:
    """Read the session document from ES and check cancelRequested.
    If true, update the session to cancelled and raise."""
    session = es_client.get(config.INDEX_SESSIONS, session_id)
    if session and session.get("cancelRequested"):
        es_client.update_doc(config.INDEX_SESSIONS, session_id, {
            "status": "cancelled",
            "cancelledAt": _now(),
            "updatedAt": _now(),
            "lastUpdatedBy": "pipeline",
        })
        raise PipelineCancelled(f"Session {session_id} cancelled")


def _cleanup_roles(session_id: str) -> None:
    """Delete any role documents already written for this session."""
    roles = es_client.search(
        config.INDEX_ROLES,
        {"term": {"sessionId": session_id}},
        size=1000,
    )
    for role in roles:
        es_client.delete_doc(config.INDEX_ROLES, role["id"])


def _validate_session(session: dict) -> dict:
    """Validate session document read from ES. Defense in depth —
    Node should have validated at creation, but the pipeline is
    the last gate before compute runs."""

    if session.get("status") != "running":
        raise ValueError(
            f"Session has status '{session.get('status')}', expected 'running'")

    params = session.get("parameters")
    if not params:
        raise ValueError("Session has no parameters")

    required_fields = [
        "filterCriteria", "maxPopulation", "noiseFilterValue",
        "universalThreshold", "coverageThreshold", "softThreshold",
        "similarityThreshold", "minGroupSize", "maxRoles",
        "outlierThreshold", "birthrightCooccurrenceThreshold",
        "roleType",
    ]
    for field in required_fields:
        if params.get(field) is None:
            raise ValueError(f"Missing required parameter: {field}")

    if params["maxPopulation"] > _MAX_POPULATION_CEILING:
        raise ValueError(
            f"maxPopulation {params['maxPopulation']} "
            f"exceeds ceiling {_MAX_POPULATION_CEILING}")

    if not params["filterCriteria"]:
        raise ValueError("filterCriteria is empty")

    if params["roleType"] not in ("birthright", "job_roles"):
        raise ValueError(
            f"Invalid roleType: '{params['roleType']}'")

    return params


def run_pipeline(session_id: str) -> None:
    """Entry point called in a background thread by POST /sessions/:id/run.

    On success, session and roles are written to ES. On failure, the
    session is marked failed and the exception re-raised. On cancellation,
    the session is marked cancelled, orphaned roles are cleaned up, and
    PipelineCancelled is raised.
    """
    session = es_client.get(config.INDEX_SESSIONS, session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")
    params = _validate_session(session)

    try:
        # Step 1 — Filter population
        population_ids, population_summary, population_assignments = (
            filter_population(params["filterCriteria"], params["maxPopulation"])
        )
        _check_cancelled(session_id)

        # Entitlement metadata — one lookup, used by Steps 3 and 6
        all_ent_ids = sorted({
            ent_id
            for ent_ids in population_assignments.values()
            for ent_id in ent_ids
        })
        ent_lookup = fetch_entitlement_metadata(all_ent_ids)

        # Step 2 — Build sparse matrix
        matrix, entitlement_index, dropped_count = build_sparse_matrix(
            population_ids, population_assignments, params["noiseFilterValue"]
        )
        _check_cancelled(session_id)

        # Step 3 — Compute universal
        layer1_roles, residual, high_residual = compute_universal(
            matrix, entitlement_index, population_ids,
            session_id, params, population_summary, ent_lookup
        )
        _check_cancelled(session_id)

        layer1_roles_sorted = sorted(
            layer1_roles,
            key=lambda r: (
                r["applications"][0]["appName"] if r["applications"] else "",
                r["id"],
            ),
        )

        if params["roleType"] == "birthright":
            for role in layer1_roles_sorted:
                es_client.index_doc(config.INDEX_ROLES, role["id"], role)

            es_client.update_doc(config.INDEX_SESSIONS, session_id, {
                "status": "complete",
                "completedAt": _now(),
                "updatedAt": _now(),
                "lastUpdatedBy": "pipeline",
                "populationSize": len(population_ids),
                "totalEntitlementsConsidered": matrix.shape[1] + dropped_count,
                "totalEntitlementsDropped": dropped_count,
                "layer1RoleIds": [r["id"] for r in layer1_roles_sorted],
                "layer1RoleCount": len(layer1_roles_sorted),
                "highResidualPrevalenceEntitlements": high_residual,
                "populationSummary": population_summary,
            })
            return

        # Step 4 — Build similarity graph
        adjacency, graph_metrics = build_similarity_graph(
            residual, params["similarityThreshold"]
        )
        _check_cancelled(session_id)

        # Step 5 — Detect communities
        communities, suppressed_communities, modularity = detect_communities(
            adjacency, population_ids,
            params["minGroupSize"], params["maxRoles"]
        )
        _check_cancelled(session_id)

        # Step 6 — Compose roles
        layer2_roles, suppressed_roles = compose_roles(
            communities, residual, entitlement_index, population_ids,
            modularity, session_id, params, population_summary, ent_lookup
        )
        _check_cancelled(session_id)

        population_summary["suppressedCommunities"] = (
            suppressed_communities + suppressed_roles
        )

        # Write phase — Layer 1 roles first, then Layer 2
        for role in layer1_roles_sorted:
            es_client.index_doc(config.INDEX_ROLES, role["id"], role)
        for role in layer2_roles:
            es_client.index_doc(config.INDEX_ROLES, role["id"], role)

        es_client.update_doc(config.INDEX_SESSIONS, session_id, {
            "status": "complete",
            "completedAt": _now(),
            "updatedAt": _now(),
            "lastUpdatedBy": "pipeline",
            "populationSize": len(population_ids),
            "totalEntitlementsConsidered": matrix.shape[1] + dropped_count,
            "totalEntitlementsDropped": dropped_count,
            "layer1RoleIds": [r["id"] for r in layer1_roles_sorted],
            "layer1RoleCount": len(layer1_roles_sorted),
            "candidateRoleIds": [r["id"] for r in layer2_roles],
            "candidateRoleCount": len(layer2_roles),
            "singletonCount": graph_metrics["singletonCount"],
            "graphMetrics": graph_metrics,
            "highResidualPrevalenceEntitlements": high_residual,
            "populationSummary": population_summary,
        })

    except PipelineCancelled:
        _cleanup_roles(session_id)
        raise

    except Exception as exc:
        es_client.update_doc(config.INDEX_SESSIONS, session_id, {
            "status": "failed",
            "errorDetail": str(exc),
            "updatedAt": _now(),
            "lastUpdatedBy": "pipeline",
        })
        raise
