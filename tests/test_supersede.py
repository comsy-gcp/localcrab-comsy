"""Superseded documents (metadata ``supersedes`` + OPENCRAB_SUPERSEDED_WEIGHT)."""

from __future__ import annotations

import copy
import logging
from typing import Any

import pytest

from opencrab.ontology.supersede import (
    MAX_SHOWN,
    MAX_SUPERSEDES,
    apply_supersession,
    normalize_supersedes,
    stored_metadata,
    superseded_by_map,
    superseded_weight,
)
from tests.test_chunking import FakeChroma, _ingest

# ---------------------------------------------------------------------------
# normalize_supersedes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("", []),
        ("old", ["old"]),
        (" old ", ["old"]),
        ("a\nb\n", ["a", "b"]),
        (["a", "b", "a", " ", "self"], ["a", "b"]),
        ([], []),
    ],
)
def test_normalize_supersedes(value: Any, expected: list[str]) -> None:
    assert normalize_supersedes(value, "self") == expected


@pytest.mark.parametrize(
    "value",
    [
        1,
        {"a": 1},
        ["a", 2],
        [None],
        ["a\nb"],  # a line break inside one id would split it when read back
        ["a\rb"],
        ["x" * 1025],
        [f"id-{i}" for i in range(MAX_SUPERSEDES + 1)],
    ],
)
def test_normalize_supersedes_rejects_invalid_values(value: Any) -> None:
    with pytest.raises(ValueError):
        normalize_supersedes(value, "self")


def test_normalize_supersedes_allows_the_maximum() -> None:
    ids = [f"id-{i}" for i in range(MAX_SUPERSEDES)]

    assert normalize_supersedes(ids + ["self"], "self") == ids


# ---------------------------------------------------------------------------
# stored_metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("metadata", [None, {}, {"space": "policy"}])
def test_metadata_without_supersedes_is_returned_as_is(metadata: Any) -> None:
    assert stored_metadata(metadata, "doc") is metadata


def test_stored_metadata_flags_and_encodes() -> None:
    meta = {"space": "policy", "supersedes": ["v1", "v0"], "has_supersedes": "spoofed"}

    out = stored_metadata(meta, "v2")

    assert out == {"space": "policy", "supersedes": "v1\nv0", "has_supersedes": True}
    assert meta["supersedes"] == ["v1", "v0"]  # input untouched


@pytest.mark.parametrize("empty", [[], "", None, ["self"]])
def test_empty_supersedes_is_stored_as_an_explicit_clear(empty: Any) -> None:
    assert stored_metadata({"supersedes": empty}, "self") == {
        "supersedes": "",
        "has_supersedes": False,
    }


# ---------------------------------------------------------------------------
# superseded_weight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("raw", "expected"), [("0.5", 0.5), (" 1 ", 1.0), ("0.01", 0.01)])
def test_valid_weights(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("OPENCRAB_SUPERSEDED_WEIGHT", raw)

    assert superseded_weight() == expected


def test_unset_weight_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_SUPERSEDED_WEIGHT", raising=False)

    assert superseded_weight() is None


@pytest.mark.parametrize("raw", ["0", "0.005", "1e-200", "1.5", "-0.5", "abc", "nan", "inf"])
def test_invalid_weight_is_off_with_one_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    from opencrab.ontology import supersede

    supersede._warn_once.cache_clear()
    monkeypatch.setenv("OPENCRAB_SUPERSEDED_WEIGHT", raw)

    with caplog.at_level(logging.WARNING):
        assert superseded_weight() is None
        assert superseded_weight() is None

    assert caplog.text.count("OPENCRAB_SUPERSEDED_WEIGHT") == 1


# ---------------------------------------------------------------------------
# superseded_by_map
# ---------------------------------------------------------------------------


def test_map_lists_every_newer_source_once() -> None:
    metadatas = [
        {"source_id": "v2", "supersedes": ["v1"]},
        {"source_id": "v2", "supersedes": ["v1"]},  # another chunk of v2
        {"source_id": "v3", "supersedes": ["v2", "v1", "v3"]},  # self-reference ignored
        {"source_id": "legacy", "supersedes": "v0"},  # a plain string is accepted
        {"source_id": "odd", "supersedes": 7},  # unusable value is skipped
        {"supersedes": ["v1"]},  # no source_id: skipped
    ]

    assert superseded_by_map(metadatas) == {"v1": ["v2", "v3"], "v2": ["v3"], "v0": ["legacy"]}


# ---------------------------------------------------------------------------
# apply_supersession
# ---------------------------------------------------------------------------


def _item(source_id: str, score: float, cross: float | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {
        "node_id": f"{source_id}#0",
        "score": score,
        "rerank_score": score,
        "metadata": {"source_id": source_id},
    }
    if cross is not None:
        item["cross_encoder_score"] = cross
    return item


def test_superseded_results_are_marked_and_moved_down() -> None:
    items = [_item("old", 0.9), _item("new", 0.8), _item("other", 0.5)]
    before = copy.deepcopy(items)

    out = apply_supersession(items, {"old": ["new"]}, weight=0.5)

    assert items == before  # input untouched
    assert [i["metadata"]["source_id"] for i in out] == ["new", "other", "old"]
    old = out[2]
    assert old["metadata"]["superseded_by"] == ["new"]
    assert old["rerank_score"] == pytest.approx(0.45)
    assert old["score"] == pytest.approx(0.45)
    assert "superseded_by" not in out[0]["metadata"]


def test_only_direct_naming_counts() -> None:
    # Every version is weighted once, by whoever names it. Naming every older
    # version (v3: [v2, v1]) puts the current one first; the old ones keep their
    # own relative order.
    items = [_item("v1", 0.9), _item("v2", 0.8), _item("v3", 0.7)]

    chained = apply_supersession(items, {"v1": ["v2"], "v2": ["v3"]}, weight=0.5)
    named = apply_supersession(items, {"v1": ["v2", "v3"], "v2": ["v3"]}, weight=0.5)

    by_id = {i["metadata"]["source_id"]: i for i in chained}
    assert by_id["v1"]["rerank_score"] == pytest.approx(0.45)
    assert by_id["v2"]["rerank_score"] == pytest.approx(0.4)
    assert [i["metadata"]["source_id"] for i in named] == ["v3", "v1", "v2"]
    assert named[1]["metadata"]["superseded_by"] == ["v2", "v3"]


def test_a_cycle_marks_both_once_in_any_order() -> None:
    a, b = _item("a", 0.8), _item("b", 0.8)

    first = apply_supersession([a, b], {"a": ["b"], "b": ["a"]}, 0.5)
    second = apply_supersession([a, b], {"b": ["a"], "a": ["b"]}, 0.5)

    assert [i["rerank_score"] for i in first] == [0.4, 0.4]
    assert first == second


def test_a_long_chain_costs_one_lookup_per_result() -> None:
    chain = {f"v{i}": [f"v{i + 1}"] for i in range(5000)}

    out = apply_supersession([_item("v0", 0.9), _item("unrelated", 0.5)], chain, weight=0.5)

    assert out[0]["metadata"]["source_id"] == "unrelated"


def test_superseded_by_is_sorted_and_capped() -> None:
    newer = [f"v{i:02d}" for i in range(MAX_SHOWN + 5, 0, -1)]

    out = apply_supersession([_item("old", 0.9)], {"old": newer}, weight=0.5)

    assert out[0]["metadata"]["superseded_by"] == sorted(newer)[:MAX_SHOWN]


def test_weight_one_marks_without_reordering() -> None:
    items = [_item("old", 0.9), _item("new", 0.8)]

    out = apply_supersession(items, {"old": ["new"]}, weight=1.0)

    assert [i["metadata"]["source_id"] for i in out] == ["old", "new"]
    assert out[0]["metadata"]["superseded_by"] == ["new"]


def test_reranked_head_and_fused_tail_are_reordered_separately() -> None:
    # Head scores are cross-encoder scores, tail scores fused ones: never mixed, even
    # when a tail score is higher than a weighted head score.
    items = [
        _item("old", 0.9, cross=0.9),
        _item("new", 0.8, cross=0.0),  # a cross score of 0.0 still belongs to the head
        _item("tail-a", 0.95),
        _item("tail-old", 0.6),
    ]

    out = apply_supersession(items, {"old": ["new"], "tail-old": ["x"]}, weight=0.5)

    assert [i["metadata"]["source_id"] for i in out] == ["new", "old", "tail-a", "tail-old"]
    assert out[1]["cross_encoder_score"] == 0.9  # the evidence signal keeps the raw score


def test_ordering_uses_the_rerank_score_when_present() -> None:
    a = {**_item("a", 0.1), "rerank_score": 0.9}
    b = {**_item("b", 0.9), "rerank_score": 0.2}
    zero = {**_item("zero", 0.99), "rerank_score": 0.0}

    out = apply_supersession([a, zero, b], {"x": ["y"], "b": ["z"]}, weight=0.5)

    assert [i["metadata"]["source_id"] for i in out] == ["a", "b", "zero"]


def test_unaffected_order_is_kept_even_when_scores_tie() -> None:
    items = [_item("b", 0.5), _item("a", 0.5), _item("c", 0.5)]

    out = apply_supersession(items, {"zzz": ["y"]}, weight=0.5)

    assert out == items


def test_nothing_superseded_returns_the_same_list() -> None:
    items = [_item("a", 0.5)]

    assert apply_supersession(items, {}, weight=0.5) is items


def test_items_without_source_id_are_left_alone() -> None:
    node = {"node_id": "n1", "score": 0.7, "metadata": {}}

    bare = {"node_id": "n2", "score": 0.6, "metadata": None}

    out = apply_supersession([node, bare, _item("old", 0.9)], {"old": ["new"]}, weight=0.5)

    assert out[0] is node
    assert out[1] is bare


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class SupersedeChroma(FakeChroma):
    """Merges metadata on upsert like Chroma: keys left out keep their old values."""

    def __init__(self) -> None:
        super().__init__()
        self.scans = 0

    def upsert_texts(
        self, texts: list[str], metadatas: list[dict[str, Any]], ids: list[str]
    ) -> list[str]:
        merged = [{**self.records.get(i, ("", {}))[1], **m} for i, m in zip(ids, metadatas)]
        return super().upsert_texts(texts, merged, ids)

    def metadatas_where(self, where: dict[str, Any]) -> list[dict[str, Any]]:
        self.scans += 1
        return [m for _, m in self.records.values() if all(m.get(k) == v for k, v in where.items())]


@pytest.fixture
def chunking_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_CHUNKING", "1")
    monkeypatch.delenv("OPENCRAB_CHUNK_MAX_CHARS", raising=False)


@pytest.mark.usefixtures("chunking_on")
def test_chunked_ingest_stores_supersedes_on_every_chunk() -> None:
    chroma = SupersedeChroma()
    meta = {"supersedes": "policy-v1"}

    _ingest(chroma, "# 개요\n개요\n\n## 세부\n세부", "policy-v2", meta)

    assert meta == {"supersedes": "policy-v1"}
    assert {m["source_id"] for _, m in chroma.records.values()} == {"policy-v2"}
    assert all(m["supersedes"] == "policy-v1" for _, m in chroma.records.values())
    assert all(m["has_supersedes"] is True for _, m in chroma.records.values())


def test_unchunked_ingest_stores_supersedes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_CHUNKING", raising=False)
    chroma = SupersedeChroma()

    _ingest(chroma, "본문", "policy-v2", {"supersedes": ["policy-v1", "policy-v2"]})

    _, meta = chroma.records["policy-v2"]
    assert meta["supersedes"] == "policy-v1"
    assert meta["has_supersedes"] is True


@pytest.mark.usefixtures("chunking_on")
def test_stored_metadata_survives_the_vector_store_and_maps_back() -> None:
    # The real store turns non-scalar values into str(); the stored form must not need that.
    from opencrab.stores.chroma_store import _sanitize_metadata

    chroma = SupersedeChroma()
    _ingest(chroma, "본문", "v3", {"supersedes": ["v1", "v2"]})

    stored = [_sanitize_metadata(m) for _, m in chroma.records.values()]

    assert stored[0]["supersedes"] == "v1\nv2"
    assert superseded_by_map(stored) == {"v1": ["v3"], "v2": ["v3"]}


@pytest.mark.usefixtures("chunking_on")
def test_ingest_without_supersedes_stores_no_such_keys() -> None:
    chroma = SupersedeChroma()

    _ingest(chroma, "본문", "doc", {"space": "policy"})

    assert all(
        "supersedes" not in m and "has_supersedes" not in m for _, m in chroma.records.values()
    )


@pytest.mark.usefixtures("chunking_on")
def test_invalid_supersedes_is_an_error_and_writes_nothing() -> None:
    chroma = SupersedeChroma()

    result = _ingest(chroma, "본문", "doc", {"supersedes": {"bad": 1}})

    assert "error" in result
    assert chroma.records == {}


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _hybrid(chroma: Any) -> Any:
    from opencrab.ontology.query import HybridQuery

    return HybridQuery(chroma, None)  # type: ignore[arg-type]


@pytest.mark.usefixtures("chunking_on")
def test_query_stage_marks_results_of_superseded_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_SUPERSEDED_WEIGHT", "0.5")
    chroma = SupersedeChroma()
    _ingest(chroma, "새 기준", "policy-v2", {"supersedes": "policy-v1"})
    hybrid = _hybrid(chroma)

    out = hybrid._apply_supersession([_item("policy-v1", 0.9), _item("policy-v2", 0.8)])

    assert [i["metadata"]["source_id"] for i in out] == ["policy-v2", "policy-v1"]
    assert out[1]["metadata"]["superseded_by"] == ["policy-v2"]


@pytest.mark.usefixtures("chunking_on")
@pytest.mark.parametrize(("second", "still_marked"), [({}, True), ({"supersedes": []}, False)])
def test_reingest_keeps_the_mark_unless_cleared_explicitly(
    monkeypatch: pytest.MonkeyPatch, second: dict[str, Any], still_marked: bool
) -> None:
    # The store keeps keys an upsert leaves out, so only an explicit empty value clears.
    monkeypatch.setenv("OPENCRAB_SUPERSEDED_WEIGHT", "0.5")
    chroma = SupersedeChroma()
    _ingest(chroma, "새 기준", "policy-v2", {"supersedes": "policy-v1"})
    _ingest(chroma, "새 기준", "policy-v2", second)

    out = _hybrid(chroma)._apply_supersession([_item("policy-v1", 0.9)])

    assert ("superseded_by" in out[0]["metadata"]) is still_marked


@pytest.mark.usefixtures("chunking_on")
def test_query_stage_is_off_without_the_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_SUPERSEDED_WEIGHT", raising=False)
    chroma = SupersedeChroma()
    _ingest(chroma, "새 기준", "policy-v2", {"supersedes": "policy-v1"})

    items = [_item("policy-v1", 0.9)]

    assert _hybrid(chroma)._apply_supersession(items) is items
    assert chroma.scans == 0


def test_query_stage_survives_a_store_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OPENCRAB_SUPERSEDED_WEIGHT", "0.5")
    chroma = SupersedeChroma()

    def broken(where: dict[str, Any]) -> list[dict[str, Any]]:
        raise RuntimeError("/secret/path")

    chroma.metadatas_where = broken  # type: ignore[method-assign]
    items = [_item("policy-v1", 0.9)]

    with caplog.at_level(logging.WARNING):
        assert _hybrid(chroma)._apply_supersession(items) is items

    assert "/secret/path" not in caplog.text


def test_newer_documents_the_subject_cannot_view_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from unittest.mock import MagicMock

    monkeypatch.setenv("OPENCRAB_SUPERSEDED_WEIGHT", "0.5")
    chroma = SupersedeChroma()
    monkeypatch.setenv("OPENCRAB_CHUNKING", "1")
    _ingest(chroma, "비공개 개정", "secret-v2", {"supersedes": "pub-v1"})
    _ingest(chroma, "공개 개정", "pub-v3", {"supersedes": "pub-v1"})
    hybrid = _hybrid(chroma)
    hybrid._rebac = MagicMock()
    hybrid._rebac.check.side_effect = lambda subject_id, permission, resource_id: MagicMock(
        granted=not resource_id.startswith("secret-v2")
    )

    out = hybrid._apply_supersession([_item("pub-v1", 0.9)], subject_id="alice")
    unfiltered = hybrid._apply_supersession([_item("pub-v1", 0.9)])

    assert out[0]["metadata"]["superseded_by"] == ["pub-v3"]
    assert unfiltered[0]["metadata"]["superseded_by"] == ["pub-v3", "secret-v2"]


def test_mcp_ingest_reports_an_invalid_supersedes_and_stores_nothing() -> None:
    from unittest.mock import MagicMock, patch

    from opencrab.mcp.tools import ontology_ingest

    chroma = SupersedeChroma()
    mongo = MagicMock(available=True)
    ctx = {"hybrid": _hybrid(chroma), "mongo": mongo}
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        out = ontology_ingest(text="본문", source_id="doc", metadata={"supersedes": {"bad": 1}})

    assert "supersedes" in out["error"]
    assert chroma.records == {}
    mongo.upsert_source.assert_not_called()


def test_query_calls_the_supersession_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.ontology.query import HybridQuery
    from tests.test_cross_encoder import DocStore, NoGraph, NoVectorChroma

    monkeypatch.delenv("OPENCRAB_RERANKER_MODEL", raising=False)
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = DocStore()
    seen: list[int] = []

    def spy(items: list[dict[str, Any]], subject_id: str | None = None) -> list[dict[str, Any]]:
        seen.append(len(items))
        return list(reversed(items))

    hybrid._apply_supersession = spy  # type: ignore[method-assign]

    results = hybrid.query("하루 식비", limit=2)

    assert seen and seen[0] >= 2
    assert results
