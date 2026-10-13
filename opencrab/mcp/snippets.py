"""
Shorter ``ontology_query`` results, with the full text one tool call away.

Chunk bodies make up most of a query result, and most returned chunks are
never quoted. With ``OPENCRAB_SNIPPET_CHARS`` set (for example ``300``), the
first ``OPENCRAB_SNIPPET_FULL_RESULTS`` results (default 3) keep their whole
text; later chunk results keep their chunk header and only the start of the
body, cut at whitespace, followed by `` …`` and a ``truncated`` field. The
``ontology_chunk_text`` tool returns full texts by ``node_id``.

Only chunk records whose text really opens with their chunk header are cut;
other results are left as they are. Unset, results are not changed.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from typing import Any

from opencrab.ontology.chunking import chunk_header

logger = logging.getLogger(__name__)

DEFAULT_FULL_RESULTS = 3
MAX_CHUNK_IDS = 10
MAX_CHUNK_ID_CHARS = 1024
ELLIPSIS = " …"
_SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")


@functools.lru_cache(maxsize=16)
def _warn_once(message: str) -> None:
    """Log each distinct configuration warning once, not on every request."""
    logger.warning(message)


def snippet_chars() -> int | None:
    """Body characters kept in a shortened result; None when shortening is off."""
    raw = os.getenv("OPENCRAB_SNIPPET_CHARS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value <= 0:
        _warn_once(f"Invalid OPENCRAB_SNIPPET_CHARS={raw!r}; result shortening off")
        return None
    return value


def snippets_enabled() -> bool:
    return snippet_chars() is not None


def full_results() -> int:
    raw = os.getenv("OPENCRAB_SNIPPET_FULL_RESULTS", "").strip()
    if not raw:
        return DEFAULT_FULL_RESULTS
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        _warn_once(f"Invalid OPENCRAB_SNIPPET_FULL_RESULTS={raw!r}; using {DEFAULT_FULL_RESULTS}")
        return DEFAULT_FULL_RESULTS
    return value


def shorten_query_result(result: Any) -> Any:
    """Return ``result`` with later chunk results cut down; the input is not modified."""
    chars = snippet_chars()
    items = result.get("results") if isinstance(result, dict) else None
    if chars is None or not isinstance(items, list):
        return result
    keep = full_results()
    shortened = [item if i < keep else _shorten(item, chars) for i, item in enumerate(items)]
    if all(new is old for new, old in zip(shortened, items)):
        return result
    which = f"Results after the first {keep}" if keep else "Results"
    note = (
        f"{which} may show only the start of their text: those carry 'truncated', and their "
        "text ends with ' …', which is not part of the document. Call ontology_chunk_text "
        "with their node_ids to read the full text before quoting them."
    )
    return {
        **result,
        "results": shortened,
        "snippets": {"note": note, "full_results": keep, "snippet_chars": chars},
    }


def _shorten(item: Any, chars: int) -> Any:
    split = _split_chunk_text(item)
    if split is None:
        return item
    header, body = split
    if _SURROGATE.search(body):  # count and cut characters as the client will receive them
        body = body.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    if len(body) <= chars:
        return item
    window = body[:chars]
    cut = max(window.rfind(" "), window.rfind("\n"))
    shown = body[: cut if cut >= chars // 2 else chars].rstrip()
    return {
        **item,
        "text": f"{header}\n\n{shown}{ELLIPSIS}",
        "truncated": {"shown_chars": len(shown), "total_chars": len(body)},
    }


def _split_chunk_text(item: Any) -> tuple[str, str] | None:
    """(header, body) of a chunk record whose text opens with its own header, else None."""
    if not isinstance(item, dict) or not isinstance(item.get("text"), str):
        return None
    metadata = item.get("metadata")
    if (
        not isinstance(metadata, dict)
        or "chunk_index" not in metadata
        or "source_id" not in metadata
    ):
        return None
    if item.get("node_id") != f"{metadata.get('source_id')}#{metadata['chunk_index']}":
        return None
    title, heading = metadata.get("doc_title"), metadata.get("heading_path", "")
    if not isinstance(title, str) or not isinstance(heading, str):
        return None
    header = chunk_header(title, tuple(heading.split(" > ")) if heading else ())
    if not item["text"].startswith(header + "\n\n"):
        return None
    return header, item["text"][len(header) + 2 :]
