"""AI authorship labels (metadata ``authored_by`` + OPENCRAB_AI_PROVENANCE)."""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from opencrab.ontology.provenance import (
    normalize_authored_by,
    provenance_block,
    stored_metadata,
)
from opencrab.ontology.query import QueryResult
from tests.test_chunking import FakeChroma, _ingest

# ---------------------------------------------------------------------------
# normalize_authored_by / stored_metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ai", "ai"),
        (" AI ", "ai"),
        ("human", "human"),
        ("Human", "human"),
        ("\tai\n", "ai"),
        ("\u3000human\u00a0", "human"),
    ],
)
def test_normalize_authored_by(value: str, expected: str) -> None:
    assert normalize_authored_by(value) == expected


@pytest.mark.parametrize("value", ["bot", "", " ", None, 1, ["ai"], "a i", "ＡＩ"])
def test_normalize_authored_by_rejects_other_values(value: Any) -> None:
    with pytest.raises(ValueError):
        normalize_authored_by(value)


@pytest.mark.parametrize("metadata", [None, {}, {"space": "policy"}])
def test_metadata_without_authored_by_is_returned_as_is(metadata: Any) -> None:
    assert stored_metadata(metadata) is metadata


def test_stored_metadata_normalizes_without_touching_the_input() -> None:
    meta = {"space": "policy", "authored_by": " AI "}

    out = stored_metadata(meta)

    assert out == {"space": "policy", "authored_by": "ai"}
    assert meta == {"space": "policy", "authored_by": " AI "}


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("chunking", ["1", ""])
def test_ingest_stores_the_label_on_every_record(
    monkeypatch: pytest.MonkeyPatch, chunking: str
) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNKING", chunking)
    chroma = FakeChroma()

    _ingest(chroma, "# 요약\n본문\n\n## 둘\n본문", "summary", {"authored_by": "AI"})

    assert chroma.records
    assert all(m["authored_by"] == "ai" for _, m in chroma.records.values())


@pytest.mark.parametrize("chunking", ["1", ""])
def test_invalid_label_is_an_error_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, chunking: str
) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNKING", chunking)
    chroma = FakeChroma()

    result = _ingest(chroma, "본문", "summary", {"authored_by": "robot"})

    assert "authored_by" in result["error"]
    assert chroma.records == {}


def test_mcp_ingest_stores_and_returns_the_normalized_label(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opencrab.mcp.tools import ontology_ingest
    from opencrab.ontology.query import HybridQuery

    monkeypatch.setenv("OPENCRAB_CHUNKING", "1")
    chroma = FakeChroma()
    mongo = MagicMock(available=True)
    ctx = {"hybrid": HybridQuery(chroma, None), "mongo": mongo}  # type: ignore[arg-type]
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        out = ontology_ingest(text="본문", source_id="doc", metadata={"authored_by": " AI "})

    assert out["metadata"]["authored_by"] == "ai"
    assert mongo.upsert_source.call_args.args[2]["authored_by"] == "ai"
    assert all(m["authored_by"] == "ai" for _, m in chroma.records.values())


@pytest.mark.parametrize("chunking", ["1", ""])
def test_ingest_with_supersedes_and_label_stores_both(
    monkeypatch: pytest.MonkeyPatch, chunking: str
) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNKING", chunking)
    chroma = FakeChroma()

    _ingest(chroma, "본문", "v2", {"authored_by": "ai", "supersedes": "v1"})

    _, meta = next(iter(chroma.records.values()))
    assert (meta["authored_by"], meta["supersedes"], meta["has_supersedes"]) == ("ai", "v1", True)


def test_mcp_ingest_reports_an_invalid_label_and_stores_nothing() -> None:
    from opencrab.mcp.tools import ontology_ingest
    from opencrab.ontology.query import HybridQuery

    chroma = FakeChroma()
    mongo = MagicMock(available=True)
    ctx = {"hybrid": HybridQuery(chroma, None), "mongo": mongo}  # type: ignore[arg-type]
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        out = ontology_ingest(text="본문", source_id="doc", metadata={"authored_by": "robot"})

    assert "authored_by" in out["error"]
    assert chroma.records == {}
    mongo.upsert_source.assert_not_called()


# ---------------------------------------------------------------------------
# provenance_block
# ---------------------------------------------------------------------------


def _result(node_id: str, authored_by: str | None, metadata: Any = None) -> QueryResult:
    meta = {"source_id": node_id} if authored_by is None else {"authored_by": authored_by}
    if metadata is not None:
        meta = metadata
    return QueryResult(source="hybrid", node_id=node_id, score=0.5, text=node_id, metadata=meta)


@pytest.fixture
def provenance_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_AI_PROVENANCE", "1")


@pytest.mark.parametrize(
    ("raw", "on"),
    [
        ("1", True),
        (" TRUE ", True),
        ("yes", True),
        ("on", True),
        ("0", False),
        ("", False),
        ("enabled", False),
    ],
)
def test_variable_values(monkeypatch: pytest.MonkeyPatch, raw: str, on: bool) -> None:
    monkeypatch.setenv("OPENCRAB_AI_PROVENANCE", raw)

    assert (provenance_block([_result("a", "ai")]) is not None) is on


def test_block_is_off_without_the_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_AI_PROVENANCE", raising=False)

    assert provenance_block([_result("a", "ai")]) is None


@pytest.mark.usefixtures("provenance_on")
def test_no_ai_results_give_no_block() -> None:
    assert provenance_block([_result("a", "human"), _result("b", None)]) is None
    assert provenance_block([]) is None


@pytest.mark.usefixtures("provenance_on")
@pytest.mark.parametrize("odd", ["bad", ["bad"], 1])
def test_results_with_unusable_metadata_count_as_unknown(odd: Any) -> None:
    results = [_result("a", "ai"), _result("b", None, metadata=odd)]

    assert provenance_block(results) == {"ai_authored": 1, "total": 2}


@pytest.mark.usefixtures("provenance_on")
def test_a_result_with_metadata_none_counts_as_unknown() -> None:
    bare = QueryResult(source="graph", node_id="n", score=0.1, text=None)
    bare.metadata = None  # type: ignore[assignment]

    assert provenance_block([_result("a", "ai"), bare]) == {"ai_authored": 1, "total": 2}


@pytest.mark.usefixtures("provenance_on")
def test_some_ai_results_are_counted() -> None:
    block = provenance_block([_result("a", "ai"), _result("b", "human"), _result("c", None)])

    assert block == {"ai_authored": 1, "total": 3}


@pytest.mark.usefixtures("provenance_on")
def test_all_ai_results_carry_a_note() -> None:
    from opencrab.ontology.provenance import ALL_AI_NOTE

    block = provenance_block([_result("a", "ai"), _result("b", "ai")])

    assert block == {"ai_authored": 2, "total": 2, "note": ALL_AI_NOTE}


@pytest.mark.usefixtures("provenance_on")
def test_stored_values_are_compared_exactly() -> None:
    # Only the normalized form written at ingest counts; other spellings are unknown.
    assert provenance_block([_result("a", "AI"), _result("b", "ai ")]) is None


# ---------------------------------------------------------------------------
# ontology_query tool
# ---------------------------------------------------------------------------


def _tool_query(results: list[QueryResult], limit: int = 10) -> dict[str, Any]:
    from opencrab.mcp.tools import ontology_query

    ctx = {"hybrid": MagicMock(), "billing": MagicMock()}
    ctx["hybrid"].query.return_value = results
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        return ontology_query("출장 승인 기한은?", limit=limit)


@pytest.mark.usefixtures("provenance_on")
def test_tool_adds_the_block_and_keeps_the_results() -> None:
    results = [_result("a", "ai"), _result("b", "human")]
    before = copy.deepcopy([r.to_dict() for r in results])

    payload = _tool_query(results)

    assert payload["provenance"] == {"ai_authored": 1, "total": 2}
    assert payload["results"] == before


@pytest.mark.usefixtures("provenance_on")
@pytest.mark.parametrize("limit", [0, -1])
def test_tool_gives_no_block_for_a_limit_below_one(limit: int) -> None:
    assert "provenance" not in _tool_query([_result("a", "ai")], limit=limit)


def test_tool_output_is_unchanged_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_AI_PROVENANCE", raising=False)

    assert "provenance" not in _tool_query([_result("a", "ai")])


@pytest.mark.usefixtures("provenance_on")
def test_compact_encoding_keeps_the_label(monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from opencrab.mcp.compact import encode_tool_result

    monkeypatch.setenv("OPENCRAB_COMPACT_RESULTS", "1")
    chunk = {
        "source": "hybrid",
        "node_id": "doc#0",
        "score": 0.9,
        "text": "문서: d\n\n본문",
        "metadata": {
            "source_id": "doc",
            "node_id": "doc#0",
            "chunk_index": 0,
            "doc_title": "d",
            "heading_path": "",
            "authored_by": "ai",
        },
        "graph_context": None,
    }

    encoded = json.loads(encode_tool_result("ontology_query", {"results": [chunk]}))

    assert encoded["results"][0]["metadata"]["authored_by"] == "ai"
