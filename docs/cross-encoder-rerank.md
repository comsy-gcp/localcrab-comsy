# Cross-Encoder Reranking

`ontology_query` fuses vector, BM25 and graph results with RRF and a lexical
cross-score (`opencrab/ontology/reranker.py`), weighted 0.7 and 0.3. Within
one result list, neighbouring ranks differ by only about 1.6% after RRF
normalization, so unless several lists agree, the lexical part often decides
the order. On Korean questions phrased differently from the documents this
demoted correct vector hits below passages that merely share a word.

`OPENCRAB_RERANKER_MODEL` adds a cross-encoder pass: after fusion (and after
the policy filter), the top results — `OPENCRAB_RERANK_TOP_K` or the requested
`limit`, whichever is larger — are scored again by a model that reads the
question and each passage together, and reordered by that score, so every
returned result carries a cross-encoder score in (0, 1). The model runs
in-process with onnxruntime; nothing is sent anywhere.

## Setup

```bash
python scripts/download_embedding_model.py bge-reranker-v2-m3
OPENCRAB_RERANKER_MODEL=bge-reranker-v2-m3 opencrab serve
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `OPENCRAB_RERANKER_MODEL` | unset (off) | `bge-reranker-v2-m3` |
| `OPENCRAB_RERANK_TOP_K` | `20` | How many fused results are rescored (at least `limit`; at most what fusion keeps, 4×`limit` clamped to 20–80) |
| `OPENCRAB_RERANKER_MODEL_DIR` | `~/.cache/opencrab/models/<model>` | Directory with `model.onnx` and `tokenizer.json` |

All are read from the process environment, not from `.env`. The download
script fetches the pinned files and checks their sha256 as for the embedding
models (see [multilingual-embedding.md](multilingual-embedding.md)). If the
files or their manifest are missing, building the query engine fails with the
download command; in the MCP server that makes every tool that needs the
stores return that error until the files are in place. The model directory is
read when a query engine is built.

## Model

`bge-reranker-v2-m3` is BAAI's multilingual reranker (Apache-2.0). The files
are not BAAI's own: they are the int8 ONNX export published by a third party,
`onnx-community/bge-reranker-v2-m3-ONNX`, pinned to commit
`6f5ff65298512715a1e669753bc754d2bc8f367b` with the sha256 of each file.

Question and passage together are truncated to 1,024 tokens, trimming the
longer side first; with a short question, chunks from chunked ingest at the
default `OPENCRAB_CHUNK_MAX_CHARS` fit (the longest of 231 Korean chunks in our
test corpus was 680 tokens with the question). Pairs are scored one at a time: the int8
model's scores shift slightly with padding, so batching would make a
passage's score depend on the other candidates.

## Results

Twenty Korean questions over eighteen short synthetic passages written for
this test (travel, leave, security and similar office rules): twelve
paraphrases that share no key words with the answer, and eight lexical traps
where another passage shares more words with the question. Chunked ingest,
`bge-m3` embeddings, chunk BM25 on:

| Pipeline | Paraphrase top-1 | Trap top-1 | MRR |
| --- | --- | --- | --- |
| Fusion only | 8/12 | 3/8 | 0.75 |
| Fusion + `bge-reranker-v2-m3` | 12/12 | 8/8 | 1.00 |

Other candidates measured on the same questions (model alone): mmarco-mMiniLMv2
17/20, gte-multilingual-reranker-base 18/20 (fp32) or 17/20 (int8);
bge-reranker-v2-m3 fp32 matched int8 at 2.7x the time.

## Cost

With `bge-reranker-v2-m3` int8 on 4 CPU cores, rescoring 20 results took
4–6 s per query on a 231-chunk Korean corpus (median chunk about 370 tokens),
and about 8–13 s when all 20 are full 1,200-character chunks. The model adds
about 1.3 GB of memory. The process caches one instance, for the current model
directory; a query engine built earlier keeps the instance it was given. The
model loads on the first query, and each instance runs one inference at a time
(onnxruntime already uses all cores, so concurrent queries wait rather than
stack memory).

If scoring raises, the fused order is returned and a warning naming only the
exception type is logged (exception text may quote passages). A wrong number
of scores or a non-finite score also falls back to the fused order, with a
generic warning.

The cross-encoder scores also drive the optional no-evidence signal; see
[evidence-signal.md](evidence-signal.md).
