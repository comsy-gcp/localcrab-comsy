# No-Evidence Signal

`ontology_query` always returns its best-ranked results, even when none of them
is about the question. A model answering from those results tends to build an
answer anyway. With `OPENCRAB_EVIDENCE_MIN_SCORE` set, the tool result gets an
`evidence` block that says whether anything relevant was found.

It needs the cross-encoder (`OPENCRAB_RERANKER_MODEL`, see
[cross-encoder-rerank.md](cross-encoder-rerank.md)): fused RRF scores have no
fixed scale, so they cannot be compared with a threshold.

```bash
OPENCRAB_RERANKER_MODEL=bge-reranker-v2-m3 OPENCRAB_EVIDENCE_MIN_SCORE=0.05 opencrab serve
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENCRAB_EVIDENCE_MIN_SCORE` | unset (off) | Lowest best cross-encoder score that counts as evidence; a number between 0 and 1, exclusive |

The variable is read from the process environment of the process that answers
queries. A value that is not a number strictly between 0 and 1 logs a warning
and leaves the signal off.

## Output

```json
"evidence": {
  "status": "none",
  "top_score": 0.0016,
  "min_score": 0.05,
  "note": "No returned passage is closely related to the question ..."
}
```

- `status` is `found` when the best cross-encoder score among the returned
  results is at least `min_score`, else `none`.
- `note` is present only with `none`. It tells the answering model to answer
  only from a passage that states the answer, and otherwise to say that the
  documents do not contain it.
- The results themselves are returned unchanged.
- A verdict needs a cross-encoder score on every returned result. There is no
  `evidence` block when the signal is off, when the cross-encoder is not
  configured, when the call passes `use_rerank=false`, when the cross-encoder
  failed for this query (the fused order is then kept), when there are no
  results (nothing matched, or the policy filter removed everything; `total: 0`
  already says so), or when `limit` is below 1.

## Choosing the Threshold

Measured with `bge-reranker-v2-m3` (best score per question). The questions
were written for this measurement, not taken from any evaluation set; a reduced
synthetic case runs as `test_real_reranker_separates_covered_from_absent_topics`
when the model is downloaded.

| Questions | Synthetic (12 chunks) | Korean policy documents (5 documents, 313 chunks) |
| --- | --- | --- |
| Answered by the documents | 0.626 – 1.000 | 0.684 – 0.998 |
| Topic not in the documents | 0.000 – 0.008 | 0.0001 – 0.0016 |
| Topic present, asked detail (reason, deadline, owner) missing | 0.126 – 0.998 | — |

`0.05` sits well between the first two groups.

## What It Does Not Catch

The signal tells whether the question's topic is covered at all. When the
documents describe the subject but not the reason, deadline or number that was
asked for, the cross-encoder still scores the passage high (0.998 for "why was X
made a mandatory reviewer" against a passage that names X as a mandatory reviewer
without a reason). Checking that a passage actually contains the asked-for
detail is left to the answering model.
