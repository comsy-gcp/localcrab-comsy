from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from opencrab.ontology.cross_encoder import (
    RERANKERS,
    CrossEncoder,
    rerank_results,
    rerank_top_k,
    reranker_from_env,
)
from opencrab.stores.onnx_embedding import (
    DOWNLOADS,
    MANIFEST_FILE,
    check_model_dir,
    resolve_model_dir,
)

# ---------------------------------------------------------------------------
# Fakes: the tokenizer encodes (query, passage) pairs; the session scores a pair
# by how many characters of the query appear in the passage.
# ---------------------------------------------------------------------------


class FakeEncoding:
    def __init__(self, ids: list[int], mask: list[int]) -> None:
        self.ids = ids
        self.attention_mask = mask
        self.type_ids = [0] * len(ids)


class FakeTokenizer:
    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def encode_batch(self, pairs: list[tuple[str, str]]) -> list[FakeEncoding]:
        self.pairs.extend(pairs)
        width = max(len(p) for _, p in pairs) + 1
        out = []
        for query, passage in pairs:
            overlap = sum(1 for ch in set(query) if ch in passage and not ch.isspace())
            ids = [overlap] + [1] * len(passage)
            out.append(
                FakeEncoding(
                    ids + [0] * (width - len(ids)), [1] * len(ids) + [0] * (width - len(ids))
                )
            )
        return out


class FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeSession:
    def __init__(self, inputs: tuple[str, ...] = ("input_ids", "attention_mask")) -> None:
        self._inputs = inputs
        self.calls: list[dict[str, Any]] = []

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode(n) for n in self._inputs]

    def run(self, output_names: Any, feed: dict[str, Any]) -> list[Any]:
        self.calls.append(feed)
        return [feed["input_ids"][:, :1].astype(np.float32) - 2.0]  # logit = overlap - 2


def _encoder(**kwargs: Any) -> tuple[CrossEncoder, FakeTokenizer, FakeSession]:
    tok, sess = FakeTokenizer(), kwargs.pop("session", FakeSession())
    return (
        CrossEncoder(
            "bge-reranker-v2-m3", model_dir="/unused", tokenizer=tok, session=sess, **kwargs
        ),
        tok,
        sess,
    )


# ---------------------------------------------------------------------------
# Presets and environment
# ---------------------------------------------------------------------------


def test_reranker_preset_is_pinned_and_downloadable() -> None:
    spec = RERANKERS["bge-reranker-v2-m3"]
    assert spec.max_length == 1024
    pinned = DOWNLOADS["bge-reranker-v2-m3"]
    assert len(pinned.revision) == 40
    assert ("onnx/model_int8.onnx", "model.onnx") in {
        (remote, local) for remote, local, _ in pinned.files
    }
    assert all(len(sha) == 64 for _, _, sha in pinned.files)
    assert pinned.dir_env == "OPENCRAB_RERANKER_MODEL_DIR"


def test_env_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENCRAB_RERANKER_MODEL", raising=False)
    assert reranker_from_env() is None

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", " bge-reranker-v2-m3 ")
    assert reranker_from_env() == "bge-reranker-v2-m3"

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", "jina-reranker")
    with pytest.raises(ValueError, match="bge-reranker-v2-m3"):
        reranker_from_env()


@pytest.mark.parametrize(
    ("raw", "expected"), [("", 20), ("abc", 20), ("0", 20), ("-4", 20), (" 12 ", 12)]
)
def test_top_k_parsing(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    monkeypatch.setenv("OPENCRAB_RERANK_TOP_K", raw)

    assert rerank_top_k() == expected


def test_reranker_dir_uses_its_own_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENCRAB_EMBEDDING_MODEL_DIR", "/embedding/only")
    monkeypatch.delenv("OPENCRAB_RERANKER_MODEL_DIR", raising=False)
    assert (
        resolve_model_dir("bge-reranker-v2-m3")
        == Path.home() / ".cache" / "opencrab" / "models" / "bge-reranker-v2-m3"
    )

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(tmp_path))
    assert resolve_model_dir("bge-reranker-v2-m3") == tmp_path


def test_missing_reranker_files_raise_with_download_hint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_embedding_model.py bge-reranker-v2-m3"):
        check_model_dir("bge-reranker-v2-m3", tmp_path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_scores_are_sigmoid_of_logits_in_input_order() -> None:
    enc, _, _ = _encoder(batch_size=2)
    passages = ["숙박비 한도는 12만 원입니다 정말로", "회의실", "숙박"]

    scores = enc.score("숙박비 한도", passages)

    for passage, score in zip(passages, scores):
        overlap = sum(1 for ch in set("숙박비 한도") if ch in passage and not ch.isspace())
        assert score == pytest.approx(1 / (1 + np.exp(-(overlap - 2.0))))
    assert scores[0] == max(scores)


def test_extreme_logits_never_give_a_zero_score() -> None:
    class Extreme(FakeSession):
        def run(self, output_names: Any, feed: dict[str, Any]) -> list[Any]:
            return [np.full((feed["input_ids"].shape[0], 1), -1000.0, dtype=np.float32)]

    enc, _, _ = _encoder(session=Extreme())

    assert enc.score("q", ["a", "b"])[0] > 0.0


def test_default_batch_is_one_pair_so_scores_do_not_depend_on_neighbours() -> None:
    enc, _, session = _encoder()

    enc.score("q", ["a", "bb", "ccc"])

    assert RERANKERS["bge-reranker-v2-m3"].batch_size == 1
    assert [call["input_ids"].shape[0] for call in session.calls] == [1, 1, 1]


def test_inference_calls_never_overlap() -> None:
    import time

    class Tracking(FakeSession):
        def __init__(self) -> None:
            super().__init__()
            self.active = 0
            self.peak = 0
            self.guard = threading.Lock()

        def run(self, output_names: Any, feed: dict[str, Any]) -> list[Any]:
            with self.guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.01)
            with self.guard:
                self.active -= 1
            return super().run(output_names, feed)

    session = Tracking()
    enc, _, _ = _encoder(session=session)
    threads = [threading.Thread(target=enc.score, args=("q", ["a", "b", "c"])) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    assert session.peak == 1


def test_scoring_batches_and_feeds_token_type_ids_when_wanted() -> None:
    session = FakeSession(inputs=("input_ids", "attention_mask", "token_type_ids"))
    enc, _, _ = _encoder(session=session, batch_size=2)

    enc.score("q", ["a", "bb", "ccc", "dddd", "eeeee"])

    assert len(session.calls) == 3
    assert set(session.calls[0]) == {"input_ids", "attention_mask", "token_type_ids"}


def test_empty_passages() -> None:
    enc, _, session = _encoder()

    assert enc.score("q", []) == []
    assert session.calls == []


def test_concurrent_first_calls_load_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import time

    import onnxruntime
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    for remote, local, _ in DOWNLOADS["bge-reranker-v2-m3"].files:
        (tmp_path / local).write_bytes(b"x")
    tok = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "a": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / MANIFEST_FILE).write_text(
        json.dumps(
            {"model": "bge-reranker-v2-m3", "revision": DOWNLOADS["bge-reranker-v2-m3"].revision}
        )
    )
    created: list[str] = []

    def slow_session(path: str, providers: Any = None) -> FakeSession:
        created.append(path)
        time.sleep(0.1)
        return FakeSession()

    monkeypatch.setattr(onnxruntime, "InferenceSession", slow_session)
    enc = CrossEncoder("bge-reranker-v2-m3", model_dir=str(tmp_path))
    errors: list[BaseException] = []
    start = threading.Barrier(5, timeout=5)

    def worker() -> None:
        try:
            start.wait()
            enc.score("a", ["a a"])
        except BaseException as exc:  # noqa: BLE001 - surfaced through `errors`
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive()

    assert errors == []
    assert len(created) == 1


# ---------------------------------------------------------------------------
# Reordering results
# ---------------------------------------------------------------------------


class StubScorer:
    def __init__(self, scores: dict[str, float], fail: bool = False) -> None:
        self.scores = scores
        self.fail = fail
        self.seen: list[list[str]] = []

    def score(self, query: str, passages: list[str]) -> list[float]:
        self.seen.append(passages)
        if self.fail:
            raise RuntimeError("model broke")
        return [self.scores[p] for p in passages]


ITEMS = [
    {"node_id": "a", "text": "A", "rerank_score": 0.9},
    {"node_id": "b", "text": "B", "rerank_score": 0.8},
    {"node_id": "c", "text": "C", "rerank_score": 0.7},
    {"node_id": "d", "text": "D", "rerank_score": 0.6},
]


def test_top_k_is_reordered_by_cross_encoder_and_the_rest_follows() -> None:
    scorer = StubScorer({"A": 0.1, "B": 0.2, "C": 0.95})

    out = rerank_results("q", ITEMS, top_k=3, scorer=scorer)

    assert [i["node_id"] for i in out] == ["c", "b", "a", "d"]
    assert out[0]["rerank_score"] == 0.95
    assert out[0]["fused_score"] == 0.7
    assert out[3] == ITEMS[3]
    assert scorer.seen == [["A", "B", "C"]]
    assert ITEMS[2]["rerank_score"] == 0.7  # input untouched


def test_scorer_failure_keeps_fused_order(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        out = rerank_results("q", ITEMS, top_k=3, scorer=StubScorer({}, fail=True))

    assert out == ITEMS
    assert "RuntimeError" in caplog.text
    assert "model broke" not in caplog.text  # exception text may quote passages


def test_items_without_text_are_scored_as_empty() -> None:
    items = [{"node_id": "x", "text": None}, {"node_id": "y", "text": "Y"}]

    out = rerank_results("q", items, top_k=5, scorer=StubScorer({"": 0.0, "Y": 0.5}))

    assert [i["node_id"] for i in out] == ["y", "x"]


# ---------------------------------------------------------------------------
# HybridQuery integration
# ---------------------------------------------------------------------------


class NoVectorChroma:
    available = False


class NoGraph:
    available = False


class DocStore:
    def list_nodes(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "node_id": "meal",
                "space": "s",
                "node_type": "Doc",
                "properties": {"text": "식비는 하루 5만 원까지 정산"},
            },
            {
                "node_id": "lodging",
                "space": "s",
                "node_type": "Doc",
                "properties": {"text": "숙박비는 1박 12만 원 상한, 하루"},
            },
        ]


def test_query_uses_the_cross_encoder_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.ontology import cross_encoder
    from opencrab.ontology.query import HybridQuery

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setattr(cross_encoder, "get_cross_encoder", lambda model: StubScorer({}))
    monkeypatch.setattr(
        cross_encoder,
        "rerank_results",
        lambda q, items, top_k, scorer: sorted(items, key=lambda i: i["node_id"] != "lodging"),
    )
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = DocStore()

    results = hybrid.query("호텔비는 하루 얼마까지 돼?", limit=2)

    assert results[0].node_id == "lodging"


def test_query_skips_the_cross_encoder_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.ontology import cross_encoder
    from opencrab.ontology.query import HybridQuery

    monkeypatch.delenv("OPENCRAB_RERANKER_MODEL", raising=False)

    def boom(model: str) -> Any:
        raise AssertionError("must not load a cross-encoder")

    monkeypatch.setattr(cross_encoder, "get_cross_encoder", boom)
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = DocStore()

    assert hybrid.query("하루 식비", limit=2)


# ---------------------------------------------------------------------------
# Real model (runs only when the model files have been downloaded)
# ---------------------------------------------------------------------------


def _real_model_ready() -> bool:
    try:
        check_model_dir("bge-reranker-v2-m3", resolve_model_dir("bge-reranker-v2-m3"))
    except (FileNotFoundError, ValueError):
        return False
    return True


@pytest.mark.skipif(not _real_model_ready(), reason="reranker model not downloaded")
def test_real_reranker_beats_a_lexical_trap() -> None:
    enc = CrossEncoder("bge-reranker-v2-m3")
    passages = [
        "출장 중 식비는 하루 5만 원까지 영수증으로 실비 정산한다.",
        "국내 출장 숙박비는 1박 12만 원을 넘을 수 없다.",
        "회의실 예약은 사내 포털에서 하루 전까지 한다.",
    ]

    scores = enc.score("호텔비는 하루 얼마까지 돼?", passages)

    assert int(np.argmax(scores)) == 1


def test_get_cross_encoder_fails_fast_without_verified_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from opencrab.ontology import cross_encoder

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(tmp_path / "missing"))
    cross_encoder.get_cross_encoder.cache_clear()
    try:
        with pytest.raises(
            FileNotFoundError, match="download_embedding_model.py bge-reranker-v2-m3"
        ):
            cross_encoder.get_cross_encoder("bge-reranker-v2-m3")
    finally:
        cross_encoder.get_cross_encoder.cache_clear()


def test_hybrid_query_construction_fails_fast(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from opencrab.ontology import cross_encoder
    from opencrab.ontology.query import HybridQuery

    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(tmp_path / "missing"))
    cross_encoder.get_cross_encoder.cache_clear()
    try:
        with pytest.raises(FileNotFoundError):
            HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    finally:
        cross_encoder.get_cross_encoder.cache_clear()


def test_unknown_model_and_empty_results() -> None:
    with pytest.raises(ValueError, match="bge-reranker-v2-m3"):
        CrossEncoder("not-a-reranker")

    assert rerank_results("q", [], top_k=5, scorer=StubScorer({})) == []


def _verified_dir(path: Path) -> Path:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    for _, local, _ in DOWNLOADS["bge-reranker-v2-m3"].files:
        (path / local).write_bytes(b"x")
    vocab = {"[PAD]": 0, "[UNK]": 1, **{f"w{i}": i + 2 for i in range(20)}}
    tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(path / "tokenizer.json"))
    (path / MANIFEST_FILE).write_text(
        json.dumps(
            {"model": "bge-reranker-v2-m3", "revision": DOWNLOADS["bge-reranker-v2-m3"].revision}
        )
    )
    return path


def test_long_question_is_truncated_instead_of_failing(tmp_path: Path) -> None:
    session = FakeSession()
    enc = CrossEncoder(
        "bge-reranker-v2-m3", model_dir=str(_verified_dir(tmp_path)), session=session
    )
    question = " ".join(f"w{i % 20}" for i in range(3000))

    scores = enc.score(question, ["w1 w2", " ".join(f"w{i % 20}" for i in range(2000))])

    assert len(scores) == 2
    assert all(call["input_ids"].shape[1] <= 1024 for call in session.calls)


def test_cache_follows_the_model_directory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from opencrab.ontology import cross_encoder

    cross_encoder.get_cross_encoder.cache_clear()
    try:
        monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(_verified_dir(tmp_path)))
        first = cross_encoder.get_cross_encoder("bge-reranker-v2-m3")
        assert first.model_dir == tmp_path

        monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(tmp_path / "missing"))
        with pytest.raises(FileNotFoundError):
            cross_encoder.get_cross_encoder("bge-reranker-v2-m3")
    finally:
        cross_encoder.get_cross_encoder.cache_clear()


def test_wrong_score_count_keeps_fused_order(caplog: pytest.LogCaptureFixture) -> None:
    class Short:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [0.9]

    with caplog.at_level(logging.WARNING):
        out = rerank_results("q", ITEMS, top_k=3, scorer=Short())

    assert out == ITEMS


def test_non_finite_scores_keep_fused_order() -> None:
    out = rerank_results(
        "q", ITEMS, top_k=3, scorer=StubScorer({"A": float("nan"), "B": 0.9, "C": 0.1})
    )

    assert out == ITEMS


def test_register_download_refuses_a_conflicting_spec() -> None:
    from opencrab.stores.onnx_embedding import register_download

    register_download("bge-reranker-v2-m3", DOWNLOADS["bge-reranker-v2-m3"])  # same spec: fine
    with pytest.raises(ValueError, match="already registered"):
        register_download("bge-m3", DOWNLOADS["bge-reranker-v2-m3"])


class ScoredDocStore:
    def list_nodes(self, limit: int = 100) -> list[dict[str, Any]]:
        return [
            {
                "node_id": f"n{i}",
                "space": "s",
                "node_type": "Doc",
                "properties": {"text": f"하루 비용 {i}"},
            }
            for i in range(4)
        ]


def test_every_returned_result_carries_a_cross_encoder_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from opencrab.ontology import cross_encoder
    from opencrab.ontology.query import HybridQuery

    class ByLastDigit:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [0.1 * (int(p.strip()[-1]) + 1) for p in passages]

    scorer = ByLastDigit()
    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setenv("OPENCRAB_RERANK_TOP_K", "1")
    monkeypatch.setattr(cross_encoder, "get_cross_encoder", lambda model: scorer)
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = ScoredDocStore()

    results = hybrid.query("하루 비용", limit=3)

    assert len(results) == 3
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
    for r in results:  # every returned score is the cross-encoder's, not a fused one
        assert r.score == pytest.approx(0.1 * (int(r.node_id[-1]) + 1))


def test_use_rerank_false_skips_the_cross_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    from opencrab.ontology import cross_encoder
    from opencrab.ontology.query import HybridQuery

    scorer = StubScorer({})
    monkeypatch.setenv("OPENCRAB_RERANKER_MODEL", "bge-reranker-v2-m3")
    monkeypatch.setattr(cross_encoder, "get_cross_encoder", lambda model: scorer)
    hybrid = HybridQuery(NoVectorChroma(), NoGraph())  # type: ignore[arg-type]
    hybrid._doc_store = ScoredDocStore()

    hybrid.query("하루 비용", limit=3, use_rerank=False)

    assert scorer.seen == []


def test_infinite_logits_fall_back_to_fused_order() -> None:
    class Infinite(FakeSession):
        def run(self, output_names: Any, feed: dict[str, Any]) -> list[Any]:
            return [np.full((feed["input_ids"].shape[0], 1), np.inf, dtype=np.float32)]

    enc, _, _ = _encoder(session=Infinite())

    assert rerank_results("q", ITEMS, top_k=3, scorer=enc) == ITEMS


def test_only_the_latest_model_directory_stays_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from opencrab.ontology import cross_encoder

    first_dir = _verified_dir(
        (tmp_path / "a").resolve() if (tmp_path / "a").mkdir() is None else tmp_path
    )
    second_dir = _verified_dir(
        (tmp_path / "b").resolve() if (tmp_path / "b").mkdir() is None else tmp_path
    )
    cross_encoder.get_cross_encoder.cache_clear()
    try:
        monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(first_dir))
        cross_encoder.get_cross_encoder("bge-reranker-v2-m3")
        monkeypatch.setenv("OPENCRAB_RERANKER_MODEL_DIR", str(second_dir))
        cross_encoder.get_cross_encoder("bge-reranker-v2-m3")

        assert cross_encoder._shared_encoder.cache_info().currsize == 1
    finally:
        cross_encoder.get_cross_encoder.cache_clear()
