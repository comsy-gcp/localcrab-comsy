"""
Superseded documents in query results.

A revised policy or a corrected decision is often ingested as a new document
while the old one stays, so both answer the same question and the old wording
can win. Ingest the new document with ``metadata["supersedes"]`` naming the
``source_id``s it replaces (a string, one id per line, or a list). It is stored
on every record of the new document as one id per line, because the vector
store keeps only scalar metadata values. An empty list (or ``None``) clears an
earlier mark; leaving the key out keeps it, as with any other metadata key.

With ``OPENCRAB_SUPERSEDED_WEIGHT`` set (for example ``0.5``), query results
from a superseded document get ``metadata["superseded_by"]`` (the newer
``source_id``s) and their score multiplied by the weight, so the current text
ranks first. Only direct naming counts: a new version should name every older
version it replaces. Cross-encoder scores used by the evidence signal are not
changed. Unset, results are not touched.
"""

from __future__ import annotations

import functools
import logging
import math
import os
from collections.abc import Iterable
from typing import Any

logger = logging.getLogger(__name__)

SUPERSEDES = "supersedes"
HAS_SUPERSEDES = "has_supersedes"  # scalar flag, so the store can filter on it
MAX_SUPERSEDES = 20
MAX_ID_CHARS = 1024
MAX_SHOWN = 10  # newer ids listed in one result's superseded_by
MIN_WEIGHT = 0.01  # lower weights could round scores to 0 and keep the old order


@functools.lru_cache(maxsize=16)
def _warn_once(message: str) -> None:
    """Log each distinct configuration warning once, not on every query."""
    logger.warning(message)


def normalize_supersedes(value: Any, source_id: str) -> list[str]:
    """``source_id``s a document replaces: no blanks, repeats or itself; ValueError if invalid."""
    if value is None:
        return []
    items = value.split("\n") if isinstance(value, str) else value
    if not isinstance(items, list) or not all(isinstance(v, str) for v in items):
        raise ValueError("supersedes must be a source_id string or a list of them")
    ids = list(dict.fromkeys(v.strip() for v in items if v.strip() and v.strip() != source_id))
    if len(ids) > MAX_SUPERSEDES:
        raise ValueError(f"supersedes may name at most {MAX_SUPERSEDES} source_ids")
    for i in ids:
        if len(i) > MAX_ID_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in i):
            raise ValueError(
                f"supersedes ids must be at most {MAX_ID_CHARS} characters without control characters"
            )
    return ids


def stored_metadata(metadata: dict[str, Any] | None, source_id: str) -> dict[str, Any] | None:
    """Metadata as stored: ``supersedes`` normalized and flagged, or cleared when empty.

    Metadata without the key is returned as it is. Raises ValueError for an
    invalid value.
    """
    if not metadata or SUPERSEDES not in metadata:
        return metadata
    replaced = normalize_supersedes(metadata[SUPERSEDES], source_id)
    rest = {k: v for k, v in metadata.items() if k not in (SUPERSEDES, HAS_SUPERSEDES)}
    # Written even when empty: the vector store keeps keys an upsert leaves out.
    return {**rest, SUPERSEDES: "\n".join(replaced), HAS_SUPERSEDES: bool(replaced)}


def superseded_weight() -> float | None:
    """Score multiplier for superseded results; None when the feature is off."""
    raw = os.getenv("OPENCRAB_SUPERSEDED_WEIGHT", "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not MIN_WEIGHT <= value <= 1.0:  # also rejects NaN
        _warn_once(f"Invalid OPENCRAB_SUPERSEDED_WEIGHT={raw!r} (need {MIN_WEIGHT} <= w <= 1); off")
        return None
    return value


def superseded_by_map(metadatas: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Map each superseded ``source_id`` to the newer ``source_id``s, from stored metadata."""
    newer: dict[str, dict[str, None]] = {}
    for meta in metadatas:
        source_id = meta.get("source_id")
        if not isinstance(source_id, str) or not source_id:
            continue
        try:
            replaced = normalize_supersedes(meta.get(SUPERSEDES), source_id)
        except ValueError:
            continue
        for old in replaced:
            newer.setdefault(old, {})[source_id] = None
    return {old: list(sources) for old, sources in newer.items()}


def apply_supersession(
    items: list[dict[str, Any]],
    superseded_by: dict[str, list[str]],
    weight: float,
) -> list[dict[str, Any]]:
    """Mark and down-weight results of superseded documents; the input is not modified.

    Results scored by the cross-encoder and the fused results after them are
    re-sorted separately, because their scores are on different scales.
    """
    if not superseded_by:
        return items
    marked = [_mark(item, superseded_by, weight) for item in items]
    if all(new is old for new, old in zip(marked, items)):
        return items
    head = [i for i in marked if "cross_encoder_score" in i]
    tail = [i for i in marked if "cross_encoder_score" not in i]
    return sorted(head, key=_effective_score, reverse=True) + sorted(
        tail, key=_effective_score, reverse=True
    )


def _mark(
    item: dict[str, Any],
    superseded_by: dict[str, list[str]],
    weight: float,
) -> dict[str, Any]:
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        return item
    source_id = metadata.get("source_id")
    newer = superseded_by.get(source_id) if isinstance(source_id, str) else None
    if not newer or not isinstance(source_id, str):
        return item
    return {
        **item,
        "metadata": {**metadata, "superseded_by": sorted(newer)[:MAX_SHOWN]},
        "score": float(item.get("score") or 0.0) * weight,
        "rerank_score": _effective_score(item) * weight,
    }


def _effective_score(item: dict[str, Any]) -> float:
    """The score results are ordered by: the rerank score when present, else the plain score."""
    score = item.get("rerank_score")
    if score is None:
        score = item.get("score")
    return float(score or 0.0)
