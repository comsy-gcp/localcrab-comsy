"""
Local cross-encoder reranking for hybrid query results.

The fusion reranker (:mod:`opencrab.ontology.reranker`) combines rank lists
with RRF, whose scores barely differ between neighbouring ranks, so its
lexical cross-score largely decides the order. On Korean paraphrase questions
that pushed a correct vector hit below chunks that merely share a word.

With ``OPENCRAB_RERANKER_MODEL`` set, the top ``OPENCRAB_RERANK_TOP_K``
(default 20) fused results are scored again by a cross-encoder that reads the
question and each passage together, and reordered by that score. The model
runs in-process with onnxruntime from files fetched once by
``scripts/download_embedding_model.py``; nothing is sent anywhere.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from opencrab.stores.onnx_embedding import (
    MODEL_FILE,
    TOKENIZER_FILE,
    DownloadSpec,
    check_model_dir,
    register_download,
    resolve_model_dir,
)

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 20
_LOGIT_CLIP = 30.0  # keeps sigmoid scores strictly inside (0, 1)


@dataclass(frozen=True)
class RerankerSpec:
    """How to run one supported cross-encoder."""

    max_length: int  # tokens for question + passage together
    batch_size: int


RERANKERS: dict[str, RerankerSpec] = {
    # BAAI/bge-reranker-v2-m3 (Apache-2.0), int8 ONNX export by onnx-community.
    # One pair per run: the int8 model's scores shift slightly with padding, so batching
    # would make a pair's score depend on its neighbours. It is not slower on CPU.
    "bge-reranker-v2-m3": RerankerSpec(max_length=1024, batch_size=1),
}

register_download(
    "bge-reranker-v2-m3",
    DownloadSpec(
        repo="onnx-community/bge-reranker-v2-m3-ONNX",
        revision="6f5ff65298512715a1e669753bc754d2bc8f367b",
        files=(
            (
                "onnx/model_int8.onnx",
                MODEL_FILE,
                "912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d",
            ),
            (
                "tokenizer.json",
                TOKENIZER_FILE,
                "8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3",
            ),
        ),
        dir_env="OPENCRAB_RERANKER_MODEL_DIR",
    ),
)


class Scorer(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


def reranker_from_env() -> str | None:
    """Return the model named by ``OPENCRAB_RERANKER_MODEL``, or None when reranking is off."""
    raw = os.getenv("OPENCRAB_RERANKER_MODEL", "").strip()
    if not raw:
        return None
    if raw not in RERANKERS:
        raise ValueError(
            f"Unknown OPENCRAB_RERANKER_MODEL={raw!r}; choose one of {', '.join(RERANKERS)}"
        )
    return raw


def rerank_top_k() -> int:
    raw = os.getenv("OPENCRAB_RERANK_TOP_K", "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_TOP_K
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("Invalid OPENCRAB_RERANK_TOP_K=%r; using %d", raw, _DEFAULT_TOP_K)
        return _DEFAULT_TOP_K
    return value


def get_cross_encoder(model: str) -> CrossEncoder:
    """Shared instance for the model's current directory; fails at once if files are missing or unverified."""
    model_dir = resolve_model_dir(model)
    check_model_dir(model, model_dir)
    return _shared_encoder(model, str(model_dir))


@functools.lru_cache(maxsize=1)  # only the current directory's model stays loaded
def _shared_encoder(model: str, model_dir: str) -> CrossEncoder:
    return CrossEncoder(model, model_dir=model_dir)


get_cross_encoder.cache_clear = _shared_encoder.cache_clear  # type: ignore[attr-defined]


class CrossEncoder:
    """Score (question, passage) pairs with a local ONNX cross-encoder."""

    def __init__(
        self,
        model: str,
        model_dir: str | None = None,
        batch_size: int | None = None,
        *,
        session: Any = None,
        tokenizer: Any = None,
    ) -> None:
        if model not in RERANKERS:
            raise ValueError(f"Unknown reranker {model!r}; choose one of {', '.join(RERANKERS)}")
        self._model = model
        self._spec = RERANKERS[model]
        self.model_dir = Path(model_dir).expanduser() if model_dir else resolve_model_dir(model)
        self._batch_size = max(1, int(batch_size or self._spec.batch_size))
        self._session = session
        self._tokenizer = tokenizer
        self._load_lock = threading.Lock()
        # One inference at a time: onnxruntime already uses every core, so parallel calls
        # only stack activation memory.
        self._run_lock = threading.Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance in (0, 1) for each passage, in input order (NaN where the model output is not finite)."""
        if not passages:
            return []
        session, tokenizer = self._load()
        order = sorted(range(len(passages)), key=lambda i: len(passages[i]))
        scores = [0.0] * len(passages)
        for start in range(0, len(order), self._batch_size):
            ids = order[start : start + self._batch_size]
            logits = self._logits(session, tokenizer, query, [passages[i] for i in ids])
            for i, logit in zip(ids, logits):
                if not math.isfinite(float(logit)):
                    scores[i] = math.nan  # rerank_results falls back to the fused order
                    continue
                clipped = min(max(float(logit), -_LOGIT_CLIP), _LOGIT_CLIP)
                scores[i] = 1.0 / (1.0 + math.exp(-clipped))
        return scores

    def _logits(self, session: Any, tokenizer: Any, query: str, batch: list[str]) -> np.ndarray:
        encodings = tokenizer.encode_batch([(query, passage) for passage in batch])
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        feed = {
            "input_ids": input_ids,
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
        }
        wanted = {i.name for i in session.get_inputs()}
        if "token_type_ids" in wanted:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)
        with self._run_lock:
            output = session.run(None, {k: v for k, v in feed.items() if k in wanted})[0]
        return np.asarray(output, dtype=np.float64).reshape(len(batch), -1)[:, 0]

    def _load(self) -> tuple[Any, Any]:
        if self._session is not None and self._tokenizer is not None:
            return self._session, self._tokenizer
        with self._load_lock:
            session, tokenizer = self._session, self._tokenizer
            if session is None or tokenizer is None:
                check_model_dir(self._model, self.model_dir)
            if tokenizer is None:
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(self.model_dir / TOKENIZER_FILE))
                # longest_first trims whichever side is longer, so an over-long question is cut too.
                tokenizer.enable_truncation(
                    max_length=self._spec.max_length, strategy="longest_first"
                )
                tokenizer.enable_padding()
            if session is None:
                import onnxruntime

                session = onnxruntime.InferenceSession(
                    str(self.model_dir / MODEL_FILE), providers=["CPUExecutionProvider"]
                )
            self._tokenizer, self._session = tokenizer, session
            return session, tokenizer


def rerank_results(
    question: str,
    items: list[dict[str, Any]],
    top_k: int,
    scorer: Scorer,
) -> list[dict[str, Any]]:
    """Reorder the first ``top_k`` fused results by cross-encoder score; keep the rest after them.

    Reranked items get the cross-encoder score as ``rerank_score`` and keep the
    fused score as ``fused_score``. On a scoring error the fused order is returned.
    """
    head, tail = items[:top_k], items[top_k:]
    if not head:
        return list(items)
    try:
        scores = scorer.score(question, [str(item.get("text") or "") for item in head])
    except Exception as exc:  # availability first: a broken reranker must not break queries
        # Type only: exception text from the tokenizer or runtime may quote passages.
        logger.warning("Cross-encoder rerank failed; keeping fused order: %s", type(exc).__name__)
        return list(items)
    if len(scores) != len(head) or not all(math.isfinite(score) for score in scores):
        logger.warning("Cross-encoder returned unusable scores; keeping fused order")
        return list(items)
    rescored = [
        {
            **item,
            "fused_score": item.get("rerank_score", item.get("score")),
            "rerank_score": score,
            "cross_encoder_score": score,  # "rerank_score" alone may be a fused score
        }
        for item, score in zip(head, scores)
    ]
    rescored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return rescored + list(tail)
