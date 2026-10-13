"""No-evidence signal for ontology_query (OPENCRAB_EVIDENCE_MIN_SCORE)."""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from opencrab.ontology.evidence import evidence_min_score, evidence_signal
from opencrab.ontology.query import QueryResult
from tests.test_cross_encoder import DocStore, NoGraph, NoVectorChroma, StubScorer


def _result(node_id: str, cross_score: float | None) -> QueryResult:
    return QueryResult(
        source="hybrid", node_id=node_id, score=0.5, text=node_id, cross_score=cross_score
    )


@pytest.fixture
def threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_EVIDENCE_MIN_SCORE", "0.05")


# ---------------------------------------------------------------------------
# Threshold parsing
# ---------------------------------------------------------------------------


def test_unset_threshold_turns_the_signal_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_EVIDENCE_MIN_SCORE", raising=False)

    assert evidence_min_score() is None
    assert evidence_signal([_result("a", 0.001)]) is None


@pytest.mark.parametrize("raw", ["", "  "])
def test_blank_threshold_is_off(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("OPENCRAB_EVIDENCE_MIN_SCORE", raw)

    assert evidence_min_score() is None


@pytest.mark.parametrize("raw", ["abc", "0", "1", "1.5", "-0.1", "nan", "inf"])
def test_invalid_threshold_is_off_with_a_warning(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, raw: str
) -> None:
    monkeypatch.setenv("OPENCRAB_EVIDENCE_MIN_SCORE", raw)

    with caplog.at_level(logging.WARNING):
        assert evidence_min_score() is None

    assert "OPENCRAB_EVIDENCE_MIN_SCORE" in caplog.text


def test_valid_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENCRAB_EVIDENCE_MIN_SCORE", " 0.05 ")

    assert evidence_min_score() == 0.05


# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("threshold")
def test_low_top_score_reports_no_evidence() -> None:
    signal = evidence_signal([_result("a", 0.003), _result("b", 0.001)])

    assert signal is not None
    assert signal["status"] == "none"
    assert signal["top_score"] == 0.003
    assert signal["min_score"] == 0.05
    assert "note" in signal


@pytest.mark.usefixtures("threshold")
def test_high_top_score_reports_evidence_found_without_a_note() -> None:
    signal = evidence_signal([_result("a", 0.2), _result("b", 0.93)])

    assert signal == {"status": "found", "top_score": 0.93, "min_score": 0.05}


@pytest.mark.usefixtures("threshold")
def test_score_equal_to_the_threshold_counts_as_found() -> None:
    signal = evidence_signal([_result("a", 0.05)])

    assert signal is not None
    assert signal["status"] == "found"


@pytest.mark.usefixtures("threshold")
def test_no_results_give_no_signal() -> None:
    # Nothing to score: an empty result (or one emptied by the policy filter) is not judged.
    assert evidence_signal([]) is None


@pytest.mark.usefixtures("threshold")
def test_non_finite_scores_give_no_signal_in_any_order() -> None:
    nan = float("nan")

    assert evidence_signal([_result("a", nan), _result("b", 0.9)]) is None
    assert evidence_signal([_result("b", 0.9), _result("a", nan)]) is None
    assert evidence_signal([_result("a", nan)]) is None


@pytest.mark.usefixtures("threshold")
def test_results_without_cross_encoder_scores_give_no_signal() -> None:
    # Fused (RRF) scores are not on the cross-encoder scale, so no verdict is possible.
    assert evidence_signal([_result("a", None), _result("b", None)]) is None


@pytest.mark.usefixtures("threshold")
def test_partly_scored_results_give_no_signal() -> None:
    # An unscored result may be the relevant one, so a low scored head proves nothing.
    assert evidence_signal([_result("a", 0.01), _result("b", None)]) is None


# ---------------------------------------------------------------------------
# Cross-encoder scores reach QueryResult
# ---------------------------------------------------------------------------


def test_rerank_results_marks_cross_encoder_scores() -> None:
    from opencrab.ontology.cross_encoder import rerank_results

    items = [
        {"node_id": "a", "text": "A", "rerank_score": 0.9},
        {"node_id": "b", "text": "B", "rerank_score": 0.8},
        {"node_id": "c", "text": "C", "rerank_score": 0.7},
    ]

    out = rerank_results("q", items, top_k=2, scorer=StubScorer({"A": 0.1, "B": 0.6}))

    assert [(i["node_id"], i.get("cross_encoder_score")) for i in out] == [
        ("b", 0.6),
        ("a", 0.1),
        ("c", None),
    ]


class KeywordScorer:
    """Scores 0.97 for passages about lodging, 0.02 for the rest."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.97 if "숙박비" in p else 0.02 for p in passages]


def _hybrid_with_scorer(monkeypatch: pytest.MonkeyPatch, scorer: Any) -> Any:
    from opencrab.ontology import cross_encoder
    from opencrab.ontology.query import HybridQuery

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setattr(cross_encoder, "get_cross_encoder", lambda model: scorer)
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = DocStore()
    return hybrid


def test_query_results_carry_cross_encoder_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    hybrid = _hybrid_with_scorer(monkeypatch, KeywordScorer())

    results = hybrid.query("하루 숙박비", limit=2)

    assert [(r.node_id, r.cross_score) for r in results] == [("lodging", 0.97), ("meal", 0.02)]
    assert "cross_score" not in results[0].to_dict()


def test_failed_cross_encoder_leaves_cross_scores_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    hybrid = _hybrid_with_scorer(monkeypatch, StubScorer({}, fail=True))

    results = hybrid.query("하루 숙박비", limit=2)

    assert results
    assert all(r.cross_score is None for r in results)


def test_every_returned_result_is_scored_when_limit_exceeds_top_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENCRAB_RERANK_TOP_K", "1")
    hybrid = _hybrid_with_scorer(monkeypatch, KeywordScorer())

    results = hybrid.query("하루 숙박비", limit=2)

    assert len(results) == 2
    assert all(r.cross_score is not None for r in results)


def test_query_without_cross_encoder_has_no_cross_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.ontology.query import HybridQuery

    monkeypatch.delenv("OPENCRAB_RERANKER_MODEL", raising=False)
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = DocStore()

    assert all(r.cross_score is None for r in hybrid.query("하루 식비", limit=2))


# ---------------------------------------------------------------------------
# ontology_query tool
# ---------------------------------------------------------------------------


def _tool_query(results: list[QueryResult], limit: int = 10) -> dict[str, Any]:
    from opencrab.mcp.tools import ontology_query

    ctx = {"hybrid": MagicMock(), "billing": MagicMock()}
    ctx["hybrid"].query.return_value = results
    with patch("opencrab.mcp.tools._get_context", return_value=ctx):
        return ontology_query("재택근무는 주 몇 회까지 허용되나?", limit=limit)


@pytest.mark.usefixtures("threshold")
def test_tool_adds_the_evidence_signal_and_keeps_the_results() -> None:
    results = [_result("a", 0.004), _result("b", 0.002)]

    payload = _tool_query(results)

    assert payload["evidence"]["status"] == "none"
    assert payload["results"] == [r.to_dict() for r in results]


@pytest.mark.usefixtures("threshold")
def test_tool_reports_found_evidence() -> None:
    payload = _tool_query([_result("a", 0.97)])

    assert payload["evidence"] == {"status": "found", "top_score": 0.97, "min_score": 0.05}


@pytest.mark.usefixtures("threshold")
@pytest.mark.parametrize("limit", [0, -1])
def test_tool_gives_no_verdict_for_a_limit_below_one(limit: int) -> None:
    assert "evidence" not in _tool_query([_result("a", 0.004)], limit=limit)


def test_tool_output_is_unchanged_when_the_signal_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_EVIDENCE_MIN_SCORE", raising=False)

    payload = _tool_query([_result("a", 0.004)])

    assert "evidence" not in payload


@pytest.mark.usefixtures("threshold")
def test_compact_encoding_keeps_the_evidence_signal(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.mcp.compact import encode_tool_result

    monkeypatch.setenv("OPENCRAB_COMPACT_RESULTS", "1")

    encoded = json.loads(encode_tool_result("ontology_query", _tool_query([_result("a", 0.004)])))

    assert encoded["evidence"]["status"] == "none"


# ---------------------------------------------------------------------------
# Real model (runs only when the model files have been downloaded)
# ---------------------------------------------------------------------------

_CHUNKS = [
    "문서: 출장 규정\n절: 숙박비\n\n국내 출장 숙박비는 1박 12만 원을 한도로 한다.",
    "문서: 보안 규정\n절: 비밀번호\n\n비밀번호는 12자 이상으로 하고 90일마다 바꾼다.",
    "문서: 휴가 규정\n절: 연차\n\n미사용 연차는 다음 해로 이월하지 않는다.",
]


def _model_ready() -> bool:
    from opencrab.ontology.cross_encoder import check_model_dir, resolve_model_dir

    try:
        check_model_dir("bge-reranker-v2-m3", resolve_model_dir("bge-reranker-v2-m3"))
    except (FileNotFoundError, ValueError):
        return False
    return True


@pytest.mark.skipif(not _model_ready(), reason="reranker model not downloaded")
def test_real_reranker_separates_covered_from_absent_topics() -> None:
    from opencrab.ontology.cross_encoder import get_cross_encoder

    encoder = get_cross_encoder("bge-reranker-v2-m3")
    covered = ["국내 출장 숙박비 한도는 얼마인가?", "비밀번호는 얼마나 자주 바꿔야 하나?"]
    absent = ["재택근무는 주 몇 회까지 허용되나?", "법인카드 한도는 얼마인가?"]

    assert all(max(encoder.score(q, _CHUNKS)) >= 0.05 for q in covered)
    assert all(max(encoder.score(q, _CHUNKS)) < 0.05 for q in absent)
