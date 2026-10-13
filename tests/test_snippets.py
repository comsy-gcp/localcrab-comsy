"""Shortened ontology_query results and the full-text tool (OPENCRAB_SNIPPET_CHARS)."""

from __future__ import annotations

import copy
import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from opencrab.mcp.compact import encode_tool_result
from opencrab.mcp.snippets import full_results, shorten_query_result, snippet_chars

HEADER = "문서: 출장 규정\n절: 출장비 > 숙박비"


def _chunk(index: int, body: str) -> dict[str, Any]:
    return {
        "source": "hybrid",
        "node_id": f"/docs/출장 규정.md#{index}",
        "score": 0.9,
        "text": f"{HEADER}\n\n{body}",
        "metadata": {
            "source_id": "/docs/출장 규정.md",
            "doc_title": "출장 규정",
            "heading_path": "출장비 > 숙박비",
            "node_id": f"/docs/출장 규정.md#{index}",
            "chunk_index": index,
            "chunk_count": 9,
        },
        "graph_context": None,
    }


LONG = "숙박비는 1박 12만 원을 한도로 한다. " * 10  # 230 characters, spaces between words
NODE = {
    "source": "graph",
    "node_id": "policy-travel",
    "score": 0.5,
    "text": "출장 정책 " * 40,
    "metadata": {"title": "출장 정책"},
    "graph_context": None,
}


def _query(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"question": "숙박비 한도", "total": len(items), "results": items}


@pytest.fixture
def snippets_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_SNIPPET_CHARS", "50")
    monkeypatch.setenv("OPENCRAB_SNIPPET_FULL_RESULTS", "2")


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def test_unset_means_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_SNIPPET_CHARS", raising=False)
    result = _query([_chunk(i, LONG) for i in range(4)])

    assert snippet_chars() is None
    assert shorten_query_result(result) is result


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "1.5"])
def test_invalid_snippet_chars_is_off_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    from opencrab.mcp import snippets

    snippets._warn_once.cache_clear()
    monkeypatch.setenv("OPENCRAB_SNIPPET_CHARS", raw)

    with caplog.at_level(logging.WARNING):
        assert snippet_chars() is None
        assert snippet_chars() is None

    assert caplog.text.count("OPENCRAB_SNIPPET_CHARS") == 1


@pytest.mark.parametrize(("raw", "expected"), [("", 3), ("0", 0), (" 5 ", 5), ("x", 3), ("-1", 3)])
def test_full_results_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    monkeypatch.setenv("OPENCRAB_SNIPPET_FULL_RESULTS", raw)

    assert full_results() == expected


# ---------------------------------------------------------------------------
# Shortening
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("snippets_on")
def test_top_results_stay_whole_and_the_rest_are_cut() -> None:
    items = [_chunk(i, LONG) for i in range(4)]
    result = _query(items)
    before = copy.deepcopy(result)

    out = shorten_query_result(result)

    assert result == before  # input untouched
    assert [r["text"] for r in out["results"][:2]] == [items[0]["text"], items[1]["text"]]
    for r in out["results"][2:]:
        header, body = r["text"].split("\n\n", 1)
        assert header == HEADER
        assert body.endswith(" …")
        shown = body[: -len(" …")]
        assert LONG.startswith(shown)
        assert r["truncated"] == {"shown_chars": len(shown), "total_chars": len(LONG)}
        assert len(shown) <= 50
    assert "ontology_chunk_text" in out["snippets"]["note"]
    assert out["snippets"]["full_results"] == 2
    assert out["snippets"]["snippet_chars"] == 50


@pytest.mark.usefixtures("snippets_on")
def test_cut_lands_on_whitespace() -> None:
    out = shorten_query_result(_query([_chunk(0, "x"), _chunk(1, "x"), _chunk(2, LONG)]))

    body = out["results"][2]["text"].split("\n\n", 1)[1]
    assert body == "숙박비는 1박 12만 원을 한도로 한다. 숙박비는 1박 12만 원을 한도로 한다. …"


def _cut_body(body: str) -> str:
    out = shorten_query_result(_query([_chunk(0, "x"), _chunk(1, "x"), _chunk(2, body)]))
    return out["results"][2]["text"].split("\n\n", 1)[1]


@pytest.mark.usefixtures("snippets_on")
@pytest.mark.parametrize(
    ("space_at", "expected"),
    [(25, "가" * 25 + " …"), (24, "가" * 24 + " " + "가" * 25 + " …")],
)
def test_whitespace_must_be_in_the_second_half_of_the_window(space_at: int, expected: str) -> None:
    # N=50: a space at index 25 (= N//2) is used; one at 24 is too early, so the cut is hard.
    body = "가" * space_at + " " + "가" * (80 - space_at)

    assert _cut_body(body) == expected


@pytest.mark.usefixtures("snippets_on")
def test_line_breaks_are_cut_points_and_trailing_spaces_are_dropped() -> None:
    assert _cut_body("가" * 30 + "\n" + "나" * 60) == "가" * 30 + " …"
    assert _cut_body("가" * 30 + "   \n" + "나" * 60) == "가" * 30 + " …"


@pytest.mark.usefixtures("snippets_on")
@pytest.mark.parametrize(("length", "cut"), [(50, False), (51, True)])
def test_a_body_of_exactly_the_limit_is_not_cut(length: int, cut: bool) -> None:
    assert _cut_body("가" * length).endswith(" …") is cut


@pytest.mark.usefixtures("snippets_on")
def test_surrogate_pairs_are_never_split() -> None:
    pair = chr(0xD83D) + chr(0xDE00)
    body = "가" * 49 + pair * 10

    shown = _cut_body(body)[: -len(" …")]

    assert shown == "가" * 49 + chr(0x1F600)
    assert all(not 0xD800 <= ord(c) <= 0xDFFF for c in shown)


@pytest.mark.usefixtures("snippets_on")
def test_text_without_whitespace_is_cut_at_the_limit() -> None:
    body = "가" * 120

    out = shorten_query_result(_query([_chunk(0, "x"), _chunk(1, "x"), _chunk(2, body)]))

    assert out["results"][2]["text"] == f"{HEADER}\n\n{'가' * 50} …"


@pytest.mark.usefixtures("snippets_on")
def test_short_bodies_are_untouched_and_no_note_is_added() -> None:
    result = _query([_chunk(i, "짧은 본문") for i in range(4)])

    out = shorten_query_result(result)

    assert out == result
    assert "snippets" not in out


@pytest.mark.usefixtures("snippets_on")
def test_non_chunk_results_and_unverified_headers_are_untouched() -> None:
    odd = _chunk(3, LONG)
    odd["text"] = "다른 머리말\n\n" + LONG  # header does not match its metadata
    renamed = _chunk(4, LONG)
    renamed["node_id"] = "other-id"  # not {source_id}#{chunk_index}

    out = shorten_query_result(_query([_chunk(0, "x"), _chunk(1, "x"), NODE, odd, renamed]))

    assert out["results"][2:] == [NODE, odd, renamed]


def test_zero_full_results_cuts_from_the_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_SNIPPET_CHARS", "50")
    monkeypatch.setenv("OPENCRAB_SNIPPET_FULL_RESULTS", "0")

    out = shorten_query_result(_query([_chunk(0, LONG)]))

    assert "truncated" in out["results"][0]
    assert out["snippets"]["note"].startswith("Results may show")


@pytest.mark.usefixtures("snippets_on")
def test_results_that_are_not_a_list_pass_through() -> None:
    assert shorten_query_result({"error": "boom"}) == {"error": "boom"}


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("snippets_on")
@pytest.mark.parametrize("compact", ["", "1"])
def test_encoding_applies_snippets_with_or_without_compact(
    monkeypatch: pytest.MonkeyPatch, compact: str
) -> None:
    monkeypatch.setenv("OPENCRAB_COMPACT_RESULTS", compact)

    encoded = json.loads(
        encode_tool_result("ontology_query", _query([_chunk(i, LONG) for i in range(3)]))
    )

    assert "truncated" in encoded["results"][2]
    if compact:  # compact still recognises the header and drops the repeated title
        assert "doc_title" not in encoded["results"][2]["metadata"]


@pytest.mark.usefixtures("snippets_on")
def test_other_tools_are_not_shortened() -> None:
    result = _query([_chunk(i, LONG) for i in range(3)])

    assert json.loads(encode_tool_result("query_bm25", result)) == result


def test_encoding_is_unchanged_when_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_SNIPPET_CHARS", raising=False)
    monkeypatch.delenv("OPENCRAB_COMPACT_RESULTS", raising=False)
    result = _query([_chunk(i, LONG) for i in range(3)])

    assert encode_tool_result("ontology_query", result) == json.dumps(
        result, ensure_ascii=True, default=str
    )


# ---------------------------------------------------------------------------
# Full-text lookup
# ---------------------------------------------------------------------------


class FakeChroma:
    """Records are chunk records unless ``metadata`` overrides their metadata."""

    available = True

    def __init__(
        self, docs: dict[str, str], metadata: dict[str, dict[str, Any]] | None = None
    ) -> None:
        self.docs = docs
        self.metadata = metadata or {}

    def get_by_id(self, doc_id: str) -> dict[str, Any] | None:
        if doc_id not in self.docs:
            return None
        meta = self.metadata.get(doc_id)
        if meta is None:
            source_id, _, index = doc_id.rpartition("#")
            meta = {"source_id": source_id, "chunk_index": int(index), "node_id": doc_id}
        return {"id": doc_id, "document": self.docs[doc_id], "metadata": meta}


class DenyRebac:
    def __init__(self, denied: set[str]) -> None:
        self.denied = denied

    def check(self, subject_id: str, permission: str, resource_id: str) -> Any:
        return MagicMock(granted=resource_id not in self.denied)


def _hybrid(
    docs: dict[str, str], rebac: Any = None, metadata: dict[str, dict[str, Any]] | None = None
) -> Any:
    from opencrab.ontology.query import HybridQuery

    hybrid = HybridQuery(FakeChroma(docs, metadata), MagicMock(available=False))  # type: ignore[arg-type]
    hybrid._rebac = rebac
    return hybrid


def test_texts_by_id_returns_found_texts_in_order() -> None:
    hybrid = _hybrid({"a#0": "본문 A", "a#1": "본문 B"})

    assert hybrid.texts_by_id(["a#1", "x", "a#0"]) == [
        {"node_id": "a#1", "text": "본문 B"},
        {"node_id": "a#0", "text": "본문 A"},
    ]


def test_texts_by_id_applies_the_policy_filter_for_a_subject() -> None:
    hybrid = _hybrid({"a#0": "본문 A", "a#1": "비공개"}, rebac=DenyRebac({"a#1"}))

    assert hybrid.texts_by_id(["a#0", "a#1"], subject_id="u1") == [
        {"node_id": "a#0", "text": "본문 A"}
    ]
    assert len(hybrid.texts_by_id(["a#0", "a#1"])) == 2  # no subject: same as ontology_query


def test_texts_by_id_skips_records_that_are_not_chunks() -> None:
    # ontology_query checks such a record under its metadata node_id, so the lookup
    # must not hand it out under the stored id.
    hybrid = _hybrid(
        {"raw-doc": "SECRET", "a#0": "본문 A"},
        metadata={"raw-doc": {"node_id": "private-node"}},
    )

    assert hybrid.texts_by_id(["raw-doc", "a#0"]) == [{"node_id": "a#0", "text": "본문 A"}]


def test_texts_by_id_skips_chunks_whose_ids_disagree() -> None:
    hybrid = _hybrid(
        {"a#0": "본문"}, metadata={"a#0": {"source_id": "a", "chunk_index": 0, "node_id": "x"}}
    )

    assert hybrid.texts_by_id(["a#0"]) == []


def _tool(
    node_ids: Any, texts: list[dict[str, str]] | None = None, **kwargs: Any
) -> tuple[dict[str, Any], MagicMock]:
    from opencrab.mcp.tools import ontology_chunk_text

    ctx = {"hybrid": MagicMock()}
    ctx["hybrid"].texts_by_id.return_value = texts or []
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        out = ontology_chunk_text(node_ids, **kwargs)
    return out, ctx["hybrid"].texts_by_id


def test_tool_returns_chunks_and_missing_ids() -> None:
    texts = [{"node_id": "a#0", "text": "본문"}]

    out, lookup = _tool(["a#0", "a#9", "a#0"], texts=texts, subject_id="u1")

    assert out == {"chunks": texts, "missing": ["a#9"]}
    lookup.assert_called_once_with(["a#0", "a#9"], subject_id="u1")  # duplicates dropped


@pytest.mark.parametrize("bad", [[], "a#0", [1], [""], ["a" * 1025], [f"a#{i}" for i in range(11)]])
def test_tool_rejects_bad_id_lists(bad: Any) -> None:
    out, lookup = _tool(bad)

    assert "error" in out
    lookup.assert_not_called()


def test_tool_hides_lookup_error_details() -> None:
    from opencrab.mcp.tools import ontology_chunk_text

    ctx = {"hybrid": MagicMock()}
    ctx["hybrid"].texts_by_id.side_effect = RuntimeError("/secret/path broke")
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        out = ontology_chunk_text(["a#0"])

    assert out == {"error": "chunk lookup failed"}


def test_tool_hides_initialisation_error_details() -> None:
    from opencrab.mcp.tools import ontology_chunk_text

    with patch(
        "opencrab.mcp.tools._get_context", side_effect=RuntimeError("/private/model failed")
    ):
        assert ontology_chunk_text(["a#0"]) == {"error": "chunk lookup failed"}


def test_tool_is_hidden_and_refused_when_snippets_are_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.mcp.tools import available_tools, dispatch_tool

    monkeypatch.delenv("OPENCRAB_SNIPPET_CHARS", raising=False)

    assert "ontology_chunk_text" not in {t["name"] for t in available_tools()}
    with pytest.raises(KeyError):
        dispatch_tool("ontology_chunk_text", {"node_ids": ["a#0"]})
    with pytest.raises(KeyError) as unknown:
        dispatch_tool("no_such_tool", {})
    assert "ontology_chunk_text" not in str(unknown.value)


def test_server_tool_list_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.mcp.server import MCPServer

    with patch("opencrab.mcp.server.get_settings"):
        server = MCPServer()

    monkeypatch.delenv("OPENCRAB_SNIPPET_CHARS", raising=False)
    off = {t["name"] for t in server._handle_tools_list({})["tools"]}
    monkeypatch.setenv("OPENCRAB_SNIPPET_CHARS", "300")
    on = {t["name"] for t in server._handle_tools_list({})["tools"]}

    assert on - off == {"ontology_chunk_text"}


@pytest.mark.usefixtures("snippets_on")
def test_tool_is_listed_and_callable_when_snippets_are_on() -> None:
    from opencrab.mcp.tools import available_tools, dispatch_tool

    assert "ontology_chunk_text" in {t["name"] for t in available_tools()}
    ctx = {"hybrid": MagicMock()}
    ctx["hybrid"].texts_by_id.return_value = []
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        assert dispatch_tool("ontology_chunk_text", {"node_ids": ["a#0"]}) == {
            "chunks": [],
            "missing": ["a#0"],
        }
