# pipeline-service/es/client.py
from elasticsearch import Elasticsearch, NotFoundError

import config

_kwargs = {}
if config.ES_API_KEY:
    _kwargs["api_key"] = config.ES_API_KEY
_client = Elasticsearch(config.ES_HOST, **_kwargs)


def get(index: str, doc_id: str) -> dict | None:
    """Fetch a document by _id. Return its _source, or None if not found."""
    try:
        resp = _client.get(index=index, id=doc_id)
    except NotFoundError:
        return None
    return resp["_source"]


def index_doc(index: str, doc_id: str, doc: dict) -> None:
    """Index (upsert) a document with an explicit _id."""
    _client.index(index=index, id=doc_id, document=doc)


def delete_doc(index: str, doc_id: str) -> None:
    """Delete a document by _id. Ignore if not found."""
    try:
        _client.delete(index=index, id=doc_id)
    except NotFoundError:
        pass


def update_doc(index: str, doc_id: str, partial: dict) -> None:
    """Partial update via the _update API. partial holds the fields to set."""
    _client.update(index=index, id=doc_id, doc=partial)


def search(index: str, query: dict, size: int = 100,
           sort: list | None = None) -> list[dict]:
    """Run a query and return the matching _source dicts (empty list if none)."""
    body = {"query": query}
    if sort is not None:
        body["sort"] = sort
    resp = _client.search(index=index, size=size, **body)
    return [hit["_source"] for hit in resp["hits"]["hits"]]


def scroll_all(index: str, query: dict,
               max_docs: int | None = None) -> list[dict]:
    """Scroll all matching documents up to max_docs (None = unlimited)."""
    results: list[dict] = []
    resp = _client.search(
        index=index, query=query,
        size=config.ES_SCROLL_SIZE, scroll=config.ES_SCROLL_TIMEOUT,
    )
    scroll_id = resp.get("_scroll_id")
    try:
        hits = resp["hits"]["hits"]
        while hits:
            for hit in hits:
                results.append(hit["_source"])
                if max_docs is not None and len(results) >= max_docs:
                    return results
            resp = _client.scroll(
                scroll_id=scroll_id, scroll=config.ES_SCROLL_TIMEOUT,
            )
            scroll_id = resp.get("_scroll_id")
            hits = resp["hits"]["hits"]
    finally:
        if scroll_id:
            _client.clear_scroll(scroll_id=scroll_id)
    return results


def count(index: str, query: dict) -> int:
    """Return the total hit count for a query."""
    return _client.count(index=index, query=query)["count"]
