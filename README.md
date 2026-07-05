*This project has been created as part of the 42 curriculum by mmouis.*

# RAG Against the Machine

## Description

A Retrieval-Augmented Generation (RAG) pipeline that answers natural-language questions about the vllm Data In (data/raw/vllm-0.10.1) codebase.

The system ingests `.py`, `.md`, and `.txt` files from the vLLM repository, chunks them using two distinct strategies (AST-aware for Python, sliding window for Markdown), builds two separate BM25 indexes (one for docs, one for code), retrieves the most relevant chunks per question, and optionally generates grounded answers with a local `Qwen3-0.6B` model.

Everything is exposed through a single CLI built with Python Fire.

---

## System Architecture

```
data/raw/vllm-0.10.1/          ← source repository
        │
        ▼
   src/chunking.py
   ├── AST chunking   (.py files)
   └── Sliding window (.md / .txt files)
        │
        ▼
   src/indexing.py
   ├── bm25_docs  index  → data/processed/bm25_docs.pkl
   └── bm25_code  index  → data/processed/bm25_code.pkl
        │
        ▼
   __main__.py  (CLI — Python Fire)
   ├── index          build and save indexes
   ├── search         single query → stdout
   ├── search_dataset batch queries → JSON file
   ├── evaluate       recall@k against ground truth
   ├── answer         single query + LLM answer → stdout
   └── answer_dataset batch queries + LLM answers → JSON file
        │
        ▼
   src/generator.py   (Qwen3-0.6B, lazy-loaded)
   src/evaluation.py  (recall@k scorer)
   src/models.py      (Pydantic data models)
```

Both indexes are searched independently for every query; results are merged and capped at k.

---

## Chunking Strategy

**Python files — AST-aware** (`chunk_python_ast`):

The file is parsed with Python's `ast` module. Each top-level function, class header, and method becomes its own chunk. Methods include their full class hierarchy in a prepended header (e.g. `# Class: Engine > Worker`). Nested classes are handled recursively. Nodes that exceed `max_chunk_size` are split further at natural boundaries (`\nclass`, `\ndef`, `\n\n`). Imports and module-level code not captured by AST nodes are collected as additional chunks. All character offsets reference the original file. If parsing fails the file falls back to character-based splitting.

**Markdown / text files — sliding window** (`chunk_file_by_chars`):

Chunks are capped at 1 200 characters of body text with 100-character overlap between consecutive chunks. A keyword-enriched header is prepended to every chunk:

```
# File: docs/quantization.md
# Topic: quantization
# Keywords: quantization compress optimize int8 fp8 ...
```

The `DOC_SYNONYMS` table expands known domain terms (e.g. `"cli"` → `"command line interface command"`) so BM25 can match related vocabulary.

**Split-point selection** (`find_smart_split_point`):

Rather than cutting at a hard character limit the function searches backwards from the limit for the nearest language-appropriate boundary. Python: `\nclass`, `\ndef`, `\n\n`, `\n`. Markdown: `\n#`, `\n##`, `\n\n`, `\n`. Maximum chunk size is 2 000 characters and is configurable via `--max_chunk_size`.

---

## Retrieval Method

**Algorithm: BM25Okapi** (`rank_bm25` library)

BM25 is a probabilistic term-frequency ranking function. It scores each chunk by how well its token distribution matches the query, with length normalization to avoid favouring longer chunks. Chunks with a score of zero are excluded from results.

**Code-aware tokenizer** (`tokenize` in `indexing.py`):

Lowercases the input, strips all punctuation except underscores, then expands every `snake_case` token into its parts. `chunk_file_by_chars` → `chunk`, `file`, `by`, `chars`, `chunk_file_by_chars`. This allows both exact identifier matches and partial word matches.

**Two-index architecture:**

Docs and code are indexed separately. At query time both indexes are searched with the same query; results are concatenated and capped at k. This prevents the larger code index from drowning out documentation results.

**Query caching** (`search_dataset`):

A `query_cache` dict deduplicates identical questions within a batch run, skipping redundant BM25 scoring passes.

---

## Instructions

### Requirements

- Python 3.10+
- `uv` package manager

### Installation

```bash
git clone <repo-url>
cd <repo>
make install
make index
make run
make eval
```


### Makefile targets

| Target | Description |
|--------|-------------|
| `make install` | Install all project dependencies using `uv sync`. |
| `make index` | Chunk the repository and build the BM25 indexes under `data/processed/`. |
| `make run` | Run retrieval on both the documentation and code public datasets and save the search results to `data/output/search_results/`. |
| `make eval` | Evaluate the generated search results against the answered datasets for both documentation and code. |
| `make debug` | Run the retrieval pipeline under the Python debugger (`pdb`) for both datasets. |
| `make lint` | Run `flake8` and `mypy` with the project's static-analysis configuration. |
| `make clean` | Remove Python cache files (`__pycache__`, `.mypy_cache`, and `*.pyc`). |
| `make fclean` | Remove generated search results and processed BM25 index files. |


### Data layout

```
data/
├── raw/
│   └── vllm-0.10.1/        ← unzip vllm-0.10.1.zip here
├── datasets/
│   ├── AnsweredQuestions/
│   │   ├── dataset_code_public.json
│   │   └── dataset_docs_public.json
│   └── UnansweredQuestions/
│       ├── dataset_code_public.json
│       └── dataset_docs_public.json
└── processed/              ← created by `make index`
```

---

## Example Usage

**1. Index the repository**

```bash
uv run python -m src index
uv run python -m src index --repo_root data/raw/vllm-0.10.1 --max_chunk_size 2000
```

```
Starting indexing for data/raw/vllm-0.10.1 with chunk size 2000...
Building index for 18432 chunks...
Docs chunks: 4201 | Code chunks: 14231
Ingestion complete! Indices saved under data/processed/
```

**2. Search with a single query**

```bash
uv run python -m src search "How does vLLM handle tensor parallelism?" --k 5
```

**3. Run retrieval over a question dataset**

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

```
Saved student_search_results to data/output/search_results/dataset_docs_public.json
```

**4. Evaluate recall**

```bash
uv run python -m src evaluate \
  data/output/search_results/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5
```

```
Evaluation Results
========================================
Questions evaluated: 150 / 150
Recall@1:  38.0
Recall@3:  61.0
Recall@5:  72.0
Recall@10: 82.0
```

**5. Answer a single question**

```bash
uv run python -m src answer "What is PagedAttention?" --k 10
```

**6. Generate answers for a full dataset**

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

---

## Performance Analysis

Recall@k is computed per question as `found_sources / total_sources`, averaged across all questions. A retrieved chunk counts as a hit if it overlaps with the ground-truth span by **at least 5%** of the ground-truth span's length and the file paths match by suffix (tolerating different repo root prefixes).

Required thresholds from the subject:

| Metric | Threshold | Status |
|---|---|---|
| Recall@5 — docs | ≥ 80% | target |
| Recall@5 — code | ≥ 50% | target |
| Indexing time | ≤ 5 min | ✓ |
| Cold start latency | ≤ 60 s | ✓ |
| Warm throughput (1000 q) | ≤ 90 s | ✓ |

*Fill in your actual Recall@5 numbers after running evaluation.*

**Docs Recall@5 was the hardest metric.** Markdown files are longer and noisier than individual Python functions. The keyword header with `DOC_SYNONYMS` expansion was added specifically to give BM25 more vocabulary surface area on documentation queries. The 100-character overlap between consecutive doc chunks prevents relevant sentences from being split across a boundary.

---

## Design Decisions

**BM25 over dense retrieval.** BM25 is deterministic, requires no GPU at index time, handles keyword-heavy technical content well, and easily meets the 5-minute indexing and 90-second throughput requirements. Dense embeddings would likely improve paraphrased queries but add significant latency and complexity.

**Two separate indexes.** Docs and code have different token distributions. Mixing them in a single index causes code identifiers to dominate scoring for documentation queries. Separate indexes with independent retrieval and a late merge keeps both types competitive.

**AST-aware chunking for Python.** Splitting at AST boundaries ensures each chunk is semantically complete — a function is never split mid-body. Class hierarchy in method headers (`# Class: Engine > Worker`) allows a query mentioning the class name to match its methods even when the class name appears only in the definition.

**Lazy import of `RAGGenerator`.** `torch` and `transformers` are only imported when `answer` or `answer_dataset` is called. `index`, `search`, and `evaluate` stay fast and CPU-only.

**Answer caching.** `answer_dataset` writes a JSON cache after each generated answer. A crash or interruption does not lose work; repeated runs skip already-answered questions.

---

## Challenges Faced

**Docs Recall@5 below threshold.** Pure character chunking on Markdown produced chunks that were too broad and too sparse for BM25. Adding the keyword-enriched header with `DOC_SYNONYMS` expansion and 100-character overlap raised recall significantly.

**Snake_case identifiers invisible to standard tokenizers.** BM25 treats `chunk_file_by_chars` as one token. A query for `"chunk file"` would not match it. The custom tokenizer expands each `snake_case` token into its parts alongside the original, enabling both exact and partial identifier matches.

**Character offset consistency in AST chunking.** Python's `ast` module gives line numbers; the evaluator compares character offsets. `get_char_offset` converts between them by summing line lengths. Care was required to ensure offsets stored on each chunk reference the original file rather than a local substring when large AST nodes are split further.

---

## Resources

**BM25**
- [rank_bm25 library](https://github.com/dorianbrown/rank_bm25)
- Gemini to explain to me

**RAG**
- https://youtu.be/_HQ2H_0Ayy0?si=clP3Kw2ClN3DlY-m
- https://youtu.be/swvzKSOEluc?si=46hpEBzhm0KbGMeW
- https://youtu.be/T-D1OfcDW1M?si=PL8b6ubhNoYCzSuT
- https://youtu.be/r0Dciuq0knU?si=8RyyuxVJcuY_CVYb
- https://youtu.be/AS_HlJbJjH8?si=ObQTqWti3DbsWSuc

**Qwen3 (generation model)**
- <https://huggingface.co/Qwen/Qwen3-0.6B>

**AI usage**

Claude (Anthropic) was used for:
- Help Fixing  `mypy` errors across the source files.
- Help To Organize this README.
- Help To Know How to Use LLM Qwen3
- And Answer On Random Questions
