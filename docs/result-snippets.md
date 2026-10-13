# Shortened Query Results

Chunk bodies make up most of an `ontology_query` result (92% of the response
text in a 12-question Korean evaluation), yet most returned chunks are never
quoted in the answer. With `OPENCRAB_SNIPPET_CHARS` set, the first few results
keep their whole text and later chunk results keep only the start of their
body. A new tool, `ontology_chunk_text`, returns the full text by `node_id`
when the model needs it.

```bash
OPENCRAB_SNIPPET_CHARS=300 opencrab serve
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENCRAB_SNIPPET_CHARS` | unset (off) | Body characters kept in a shortened result (positive integer) |
| `OPENCRAB_SNIPPET_FULL_RESULTS` | `3` | How many leading results keep their whole text (`0` shortens all) |

Both are read from the process environment of the process that answers queries.
An invalid `OPENCRAB_SNIPPET_CHARS` logs a warning (once per value) and
leaves shortening off; an invalid `OPENCRAB_SNIPPET_FULL_RESULTS` logs a
warning and uses `3`.

## What Changes

- Only chunk records from chunked ingest (see
  [ingest-chunking.md](ingest-chunking.md)) whose text opens with their own
  `문서:`/`절:` header are shortened. Graph nodes and other results are left as
  they are.
- A shortened result keeps its header and the first `OPENCRAB_SNIPPET_CHARS`
  body characters, cut back to the last space or line break when one falls in
  the second half of that window, followed by ` …` (not part of the document).
  Surrogate pairs are joined first, so a character is never split. It gains
  `"truncated": {"shown_chars": …, "total_chars": …}`. A body that already fits
  is not changed.
- When anything was shortened, the result gets a top-level `snippets` block
  with a note pointing to `ontology_chunk_text`.
- Shortening happens when the result is encoded for the MCP client, with or
  without `OPENCRAB_COMPACT_RESULTS`. The `ontology_query` function itself still
  returns full texts.

## `ontology_chunk_text`

```json
{"node_ids": ["/docs/policy.md#12", "/docs/policy.md#13"], "subject_id": "u1"}
```

returns `{"chunks": [{"node_id": …, "text": …}], "missing": […]}`.

- Only chunk records are returned: records whose stored id, metadata `node_id`
  and `{source_id}#{chunk_index}` agree. Other records, such as documents
  ingested without chunking, are reported as `missing`.
- 1 to 10 ids per call, each a non-empty string of at most 1,024 characters;
  duplicates are ignored. Anything else is an error. A failed lookup returns
  a generic error without details.
- With `subject_id`, the same ReBAC view filter as `ontology_query` applies,
  checked against the same resource id (the chunk id). Without one, nothing is
  filtered, as with `ontology_query`; the tool only adds a way to read a chunk
  by id. Ids that do not exist, are not chunk records or may not be viewed are
  all reported in `missing`, so the response does not reveal which is which.
- The tool is listed and callable only while `OPENCRAB_SNIPPET_CHARS` is set.
  The tool list is built per request, so the list and calls follow the same
  environment. Unset, the tool list is the same as before.

## Choosing the Values

From the same evaluation run: 67 quotes in the 12 answers were found in the
returned chunk bodies. The two measures below differ: "starts within" counts a
quote whose first characters fall inside the snippet, "visible" counts a quote
that is entirely inside a whole result or inside a snippet.

- 75% of the quotes came from the top three results of their query.
- 58% of the quotes start within the first 300 body characters, so shortening
  every result would send many more lookups.

| Kept whole + snippet length | Body text sent | Quotes visible without a lookup |
| --- | --- | --- |
| Top 3 + 300 characters | 59% | 87% |
| Top 3 + 500 characters | 74% | 94% |
| Top 5 + 300 characters | 72% | 94% |

These are observations from one run, not guarantees.
