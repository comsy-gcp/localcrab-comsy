"""
No-evidence signal for hybrid query results.

A query always returns its best-ranked chunks, even when none of them is about
the question, and an answering model tends to build an answer from whatever it
is given. With ``OPENCRAB_EVIDENCE_MIN_SCORE`` set (for example ``0.05``),
``ontology_query`` adds an ``evidence`` block: ``status`` is ``none`` when the
best cross-encoder score among the results is below the threshold, else
``found``. Unless every returned result has a cross-encoder score (no
results, reranker off or failed) there is no block.

Only cross-encoder scores are used (``OPENCRAB_RERANKER_MODEL``); fused RRF
scores have no fixed scale, so without them no verdict is given. The signal
tells whether the question's topic is covered at all. It cannot tell whether a
covered topic states the reason, date or number that was asked for.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from typing import Any, Protocol

logger = logging.getLogger(__name__)

NO_EVIDENCE_NOTE = (
    "No returned passage is closely related to the question (relevance below min_score), "
    "so the stored documents probably do not cover it. Answer only from a passage that "
    "states the answer directly; otherwise say the documents do not contain it."
)


class _Scored(Protocol):
    cross_score: float | None


def evidence_min_score() -> float | None:
    """Threshold from ``OPENCRAB_EVIDENCE_MIN_SCORE``; None when unset or invalid (signal off)."""
    raw = os.getenv("OPENCRAB_EVIDENCE_MIN_SCORE", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0.0 < value < 1.0:  # also rejects NaN
        logger.warning(
            "Invalid OPENCRAB_EVIDENCE_MIN_SCORE=%r (need a number between 0 and 1); "
            "evidence signal off",
            raw,
        )
        return None
    return value


def evidence_signal(results: Sequence[_Scored]) -> dict[str, Any] | None:
    """The ``evidence`` block for a query's results, or None when it is off or has no basis."""
    min_score = evidence_min_score()
    if min_score is None:
        return None
    # Judge only when every returned result was scored: no results (nothing found, or all
    # filtered by policy) or unscored ones leave nothing reliable to judge by.
    scores: list[float] = []
    for result in results:
        score = result.cross_score
        if score is None or not math.isfinite(score):
            return None
        scores.append(score)
    if not scores:
        return None
    top = max(scores)
    if top >= min_score:
        return {"status": "found", "top_score": top, "min_score": min_score}
    return {"status": "none", "top_score": top, "min_score": min_score, "note": NO_EVIDENCE_NOTE}
