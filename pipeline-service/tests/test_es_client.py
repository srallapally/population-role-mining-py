# pipeline-service/tests/test_es_client.py
import pytest

import config
from es import client as es_client

pytestmark = pytest.mark.integration

_SCRATCH = "test_es_client_scratch"


def _refresh(index: str) -> None:
    es_client._client.indices.refresh(index=index)


@pytest.fixture(scope="module")
def scratch_index():
    """A dedicated scratch index for ES client tests — kept separate
    from the seed indexes. Created here, deleted on teardown."""
    es = es_client._client
    es.indices.delete(index=_SCRATCH, ignore_unavailable=True)
    es.indices.create(index=_SCRATCH, mappings={
        "properties": {
            "color": {"type": "keyword"},
            "batch": {"type": "keyword"},
            "group": {"type": "keyword"},
        }
    })
    yield _SCRATCH
    es.indices.delete(index=_SCRATCH, ignore_unavailable=True)


def test_get_existing(scratch_index):
    es_client.index_doc(scratch_index, "doc-1", {"field": "value"})
    _refresh(scratch_index)
    assert es_client.get(scratch_index, "doc-1") == {"field": "value"}


def test_get_not_found(scratch_index):
    assert es_client.get(scratch_index, "nonexistent") is None


def test_index_doc_creates(scratch_index):
    es_client.index_doc(scratch_index, "doc-2", {"a": 1})
    _refresh(scratch_index)
    assert es_client.get(scratch_index, "doc-2") == {"a": 1}


def test_index_doc_upserts(scratch_index):
    es_client.index_doc(scratch_index, "doc-3", {"a": 1})
    es_client.index_doc(scratch_index, "doc-3", {"a": 2, "b": 3})
    _refresh(scratch_index)
    assert es_client.get(scratch_index, "doc-3") == {"a": 2, "b": 3}


def test_update_doc_partial(scratch_index):
    es_client.index_doc(scratch_index, "doc-4", {"a": 1, "b": 2})
    es_client.update_doc(scratch_index, "doc-4", {"b": 99})
    _refresh(scratch_index)
    assert es_client.get(scratch_index, "doc-4") == {"a": 1, "b": 99}


def test_search_matches(scratch_index):
    es_client.index_doc(scratch_index, "search-1", {"color": "red"})
    es_client.index_doc(scratch_index, "search-2", {"color": "blue"})
    es_client.index_doc(scratch_index, "search-3", {"color": "red"})
    _refresh(scratch_index)
    results = es_client.search(scratch_index, {"term": {"color": "red"}})
    assert len(results) == 2


def test_search_no_matches(scratch_index):
    results = es_client.search(scratch_index, {"term": {"color": "green"}})
    assert results == []


def test_scroll_all_complete(scratch_index, monkeypatch):
    # Force a small page size so the 25 docs span multiple scroll pages.
    monkeypatch.setattr(config, "ES_SCROLL_SIZE", 10)
    for i in range(25):
        es_client.index_doc(scratch_index, f"scroll-{i}", {"batch": "scroll-test"})
    _refresh(scratch_index)
    docs = es_client.scroll_all(scratch_index, {"term": {"batch": "scroll-test"}})
    assert len(docs) == 25


def test_scroll_all_max_docs(scratch_index):
    # Re-index the 25 docs so this test does not depend on test order.
    for i in range(25):
        es_client.index_doc(scratch_index, f"scroll-{i}", {"batch": "scroll-test"})
    _refresh(scratch_index)
    docs = es_client.scroll_all(
        scratch_index, {"term": {"batch": "scroll-test"}}, max_docs=10
    )
    assert len(docs) == 10


def test_count(scratch_index):
    for i in range(5):
        es_client.index_doc(scratch_index, f"count-{i}", {"group": "count-test"})
    _refresh(scratch_index)
    assert es_client.count(scratch_index, {"term": {"group": "count-test"}}) == 5


def test_delete_doc_existing(scratch_index):
    es_client.index_doc(scratch_index, "doc-del", {"x": 1})
    _refresh(scratch_index)
    es_client.delete_doc(scratch_index, "doc-del")
    _refresh(scratch_index)
    assert es_client.get(scratch_index, "doc-del") is None


def test_delete_doc_not_found(scratch_index):
    es_client.delete_doc(scratch_index, "nonexistent")  # no exception raised
