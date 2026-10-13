# Superseded Documents

A revised policy or a corrected decision is often ingested as a new document
while the old one stays in the store. Both then answer the same question, and
the old wording can rank first: with a 2025 and a 2026 version of the same
lodging rule ingested side by side, the cross-encoder scored both 0.999 and
returned the 2025 text first.

Ingest the new document with `supersedes`, naming the `source_id`s it replaces:

```json
{"text": "...", "source_id": "travel-2026", "metadata": {"supersedes": ["travel-2025"]}}
```

and turn the query side on:

```bash
OPENCRAB_SUPERSEDED_WEIGHT=0.5 opencrab serve
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENCRAB_SUPERSEDED_WEIGHT` | unset (off) | Multiplier for the score of results from superseded documents, `0.01 <= w <= 1`; `1` marks without reordering |

The variable is read from the process environment of the process that answers
queries. An invalid value logs a warning (once per value) and leaves the
feature off.

## Ingest

- `supersedes` is set through the `metadata` of `ontology_ingest` (MCP) or
  `HybridQuery.ingest`. The `opencrab ingest` CLI has no option for it.
- The value is a `source_id`, several separated by line breaks, or a list of
  them: at most 20 ids, each at most 1,024 characters without control
  characters. Blanks, repeats and the document's own id are dropped. Any other
  value is an error, and nothing is written to any store.
- It is stored on every record of the new document as one id per line
  (`"v1\nv0"`), with a `has_supersedes` flag, because the vector store keeps
  only scalar metadata. Results from the new document show both as they are
  stored.
- An empty value (`[]`, `""` or `null`) clears an earlier mark. Re-ingesting
  without the key keeps it, as with any other metadata key, because the vector
  store keeps keys an upsert leaves out. Deleting the new document removes the
  mark; re-ingesting or deleting the old document does not.
- Ingesting with `supersedes` works whether or not the variable is set, so a
  caller that already used a `supersedes` key for something else now has it
  validated and stored in this form.

## Query

With the variable set, after the cross-encoder stage:

- Each result whose `metadata.source_id` is superseded gets
  `metadata.superseded_by` (the newer `source_id`s, sorted, at most 10) and
  its score multiplied by the weight, once. Only direct naming counts: when v3
  names only v2, v1 is still marked through v2 but v3 is not listed for it.
  Name every older version in the newest one (`"supersedes": ["v2", "v1"]`) so
  that it ranks above all of them; the old versions keep their own relative
  order. Two documents that name each other are both marked and weighted once.
- When nothing is superseded the results are returned as they are. Otherwise
  results are re-sorted by the weighted score with a stable sort: those scored
  by the cross-encoder among themselves, then the fused results after them,
  since the two scales differ. Unaffected results keep their order relative to
  each other, as each group is already sorted by that score.
- With `subject_id`, only newer documents the subject may view count (the same
  ReBAC view check as the query, on each newer record's `node_id`, else its
  `source_id`). A result never names, or loses rank to, a document the subject
  cannot see.
- The cross-encoder score used by the no-evidence signal
  ([evidence-signal.md](evidence-signal.md)) is not weighted: an old version
  still shows that the topic is covered.
- Superseded results are kept, not removed, so the change history stays
  visible. Results without a `source_id` (graph nodes) are left alone.
- The newer documents are read on every query with a metadata filter on
  `has_supersedes`, so the cost grows with the number of records of documents
  that use `supersedes`. If that read fails, results are returned unchanged
  and a warning naming only the exception type is logged.

## Trust

Results from a superseding document show its `supersedes` value to anyone who
may view that document, including ids of old documents they may not view: the
author of the new document chose to name them.

Anyone who can ingest can mark any document as superseded and lower its rank,
without touching it. Ingest has no authorization of its own, and the same
caller can already overwrite a document by reusing its `source_id`; this
feature does not widen who is trusted, but its effect is less visible. Keep
ingest limited to trusted callers, and turn the variable on only where that
holds.
