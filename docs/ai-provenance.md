# AI Authorship Labels

AI-written notes and summaries are often ingested next to the originals they
summarize. In one synthetic test (a travel rule split into four sections, plus
a one-chunk AI summary of it that got one deadline wrong), the original ranked
first for all six questions, but the summary came second every time: the
answering model saw two passages that disagree. A label tells it which passage
was declared AI-written.

Ingest with `authored_by`:

```json
{"text": "...", "source_id": "travel-summary", "metadata": {"authored_by": "ai"}}
```

and, optionally, turn on the summary block:

```bash
OPENCRAB_AI_PROVENANCE=1 opencrab serve
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENCRAB_AI_PROVENANCE` | off | `1`/`true`/`yes`/`on` adds a `provenance` block to `ontology_query` results |

## Ingest

- `authored_by` is `"human"` or `"ai"`, case and surrounding whitespace
  ignored, set through the `metadata` of `ontology_ingest` (MCP) or
  `HybridQuery.ingest`. The `opencrab ingest` CLI has no option for it. Any
  other value is an error, and nothing is written to any store. This applies
  whether or not the variable is set, so a caller that already used an
  `authored_by` key for something else now has it validated.
- The normalized value is stored on every vector-store record of the document
  and returned in each result's `metadata`. Through `ontology_ingest` it is
  also what the document store and the ingest response carry. Without the key
  nothing is stored, and the result's authorship is unknown.
- Re-ingesting without the key keeps an earlier label, as with any other
  metadata key, because the vector store keeps keys an upsert leaves out; set
  the other value to change it.

## Query

With the variable set, when at least one returned result has
`authored_by: "ai"`:

```json
"provenance": {"ai_authored": 1, "total": 3}
```

- `ai_authored` counts the returned results labeled `ai`; `total` is the number
  of returned results (after the policy filter).
- When every result is labeled `ai`, the block also has a `note` saying that
  no passage labeled as an original was among the results.
- No block when no result is labeled `ai`, or when `limit` is below 1. Ranking
  is not changed: in the test above the original already ranked first, so a
  score penalty had nothing to fix.

## Trust

The label is what the caller declared at ingest; nothing checks who or what
wrote the text. Anyone who can ingest can label an AI summary `human` or an
original `ai`, and `human` does not mean the passage is an original document.
Ingest has no authorization of its own (see the Trust section of
[supersede.md](supersede.md)); keep it limited to trusted callers, and read the
label as a hint, not a guarantee.
