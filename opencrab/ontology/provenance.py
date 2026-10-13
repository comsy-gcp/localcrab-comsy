"""
AI authorship labels on ingested documents.

AI-written notes and summaries are increasingly ingested next to the originals
they summarize. They tend to rank just below the original and can repeat it
with mistakes, so the answering model needs to know which passage is which.

Ingest with ``metadata["authored_by"]`` set to ``"human"`` or ``"ai"``; the
label is stored on every record and returned with each result's metadata.
With ``OPENCRAB_AI_PROVENANCE`` set, ``ontology_query`` adds a ``provenance``
block counting AI-written results, with a note when every result is
AI-written. Ranking is not changed.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any, Protocol

AUTHORED_BY = "authored_by"
AI = "ai"
HUMAN = "human"
_TRUTHY = {"1", "true", "yes", "on"}

ALL_AI_NOTE = (
    "Every returned passage is labeled AI-written (authored_by: ai); no passage "
    "labeled as an original was among the results."
)


class _WithMetadata(Protocol):
    metadata: dict[str, Any]


def normalize_authored_by(value: Any) -> str:
    """``"human"`` or ``"ai"``; ValueError for anything else."""
    label = value.strip().lower() if isinstance(value, str) else None
    if label not in (HUMAN, AI):
        raise ValueError('authored_by must be "human" or "ai"')
    return label


def stored_metadata(metadata: dict[str, Any] | None) -> dict[str, Any] | None:
    """Metadata as stored, with ``authored_by`` normalized; returned as is without the key."""
    if not metadata or AUTHORED_BY not in metadata:
        return metadata
    return {**metadata, AUTHORED_BY: normalize_authored_by(metadata[AUTHORED_BY])}


def provenance_enabled() -> bool:
    """True when ``OPENCRAB_AI_PROVENANCE`` asks for the ``provenance`` block."""
    return os.getenv("OPENCRAB_AI_PROVENANCE", "").strip().lower() in _TRUTHY


def provenance_block(results: Sequence[_WithMetadata]) -> dict[str, Any] | None:
    """The ``provenance`` block for query results, or None when off or nothing is AI-written."""
    if not provenance_enabled() or not results:
        return None
    ai_authored = sum(
        1 for r in results if isinstance(r.metadata, dict) and r.metadata.get(AUTHORED_BY) == AI
    )
    if not ai_authored:
        return None
    block: dict[str, Any] = {"ai_authored": ai_authored, "total": len(results)}
    if ai_authored == len(results):
        block["note"] = ALL_AI_NOTE
    return block
