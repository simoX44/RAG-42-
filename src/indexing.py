import pickle
import string
from typing import List, Any
from pathlib import Path
from rank_bm25 import BM25Okapi
from src.chunking import Chunk

INDEX_DIR = Path("data/processed")


def tokenize(text: str) -> List[str]:
    """Tokenize text for BM25 indexing, preserving snake_case sub-tokens.

    Lowercases the input, strips all punctuation except underscores, and splits
    on whitespace.  For each token that contains an underscore the individual
    parts are appended as additional tokens so that both ``my_func`` and its
    components ``my`` and ``func`` are searchable.

    Args:
        text: Raw text to tokenize.

    Returns:
        A list of lowercase string tokens, with snake_case tokens expanded
        into their constituent parts.
    """
    text = text.lower()
    punct = string.punctuation.replace('_', '')
    clean_text = text.translate(str.maketrans(punct, ' ' * len(punct)))
    tokens = clean_text.split()
    extended_tokens = []
    for t in tokens:
        extended_tokens.append(t)
        if '_' in t:
            extended_tokens.extend(t.split('_'))
    return extended_tokens


def build_and_save_index(chunks: List[Chunk]) -> None:
    """Build separate BM25 indexes for docs and code chunks and persist them.

    Splits ``chunks`` by kind (``.md``/``.txt`` vs ``.py``), builds a
    ``BM25Okapi`` index for each group, and writes four pickle files to
    ``INDEX_DIR``: ``bm25_docs.pkl``, ``chunks_docs.pkl``, ``bm25_code.pkl``,
    and ``chunks_code.pkl``.

    Args:
        chunks: All chunks produced by the chunking stage.

    Returns:
        None.  Progress is printed to stdout and indexes are written to disk.
    """
    print(f"Building index for {len(chunks)} chunks...")

    docs_chunks = [c for c in chunks if c.kind in (".md", ".txt")]
    code_chunks = [c for c in chunks if c.kind == ".py"]

    print(f"Docs chunks: {len(docs_chunks)} | Code chunks: {len(code_chunks)}")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Building docs index...")
    docs_corpus = [tokenize(c.text) for c in docs_chunks]
    bm25_docs = BM25Okapi(docs_corpus)

    with open(INDEX_DIR / "bm25_docs.pkl", "wb") as f:
        pickle.dump(bm25_docs, f)
    with open(INDEX_DIR / "chunks_docs.pkl", "wb") as f:
        pickle.dump(docs_chunks, f)

    print("Building code index...")
    code_corpus = [tokenize(c.text) for c in code_chunks]
    bm25_code = BM25Okapi(code_corpus)

    with open(INDEX_DIR / "bm25_code.pkl", "wb") as f:
        pickle.dump(bm25_code, f)
    with open(INDEX_DIR / "chunks_code.pkl", "wb") as f:
        pickle.dump(code_chunks, f)

    print(f"Indexes saved to {INDEX_DIR}")


def load_index(index_type: str = "all") -> Any:
    """Load one or both BM25 indexes from disk.

    Args:
        index_type: Which index to load.  One of:

            * ``"docs"`` — returns ``(bm25_docs, chunks_docs)``.
            * ``"code"`` — returns ``(bm25_code, chunks_code)``.
            * any other value (default ``"all"``) — returns
              ``(bm25_docs, chunks_docs, bm25_code, chunks_code)``.

    Returns:
        A tuple whose shape depends on ``index_type`` as described above.

    Raises:
        FileNotFoundError: If the requested index files are not present in
            ``INDEX_DIR``.
    """
    try:
        if index_type == "docs":
            with open(INDEX_DIR / "bm25_docs.pkl", "rb") as f:
                bm25 = pickle.load(f)
            with open(INDEX_DIR / "chunks_docs.pkl", "rb") as f:
                chunks = pickle.load(f)
            return bm25, chunks

        elif index_type == "code":
            with open(INDEX_DIR / "bm25_code.pkl", "rb") as f:
                bm25 = pickle.load(f)
            with open(INDEX_DIR / "chunks_code.pkl", "rb") as f:
                chunks = pickle.load(f)
            return bm25, chunks

        else:
            with open(INDEX_DIR / "bm25_docs.pkl", "rb") as f:
                bm25_docs = pickle.load(f)
            with open(INDEX_DIR / "chunks_docs.pkl", "rb") as f:
                chunks_docs = pickle.load(f)
            with open(INDEX_DIR / "bm25_code.pkl", "rb") as f:
                bm25_code = pickle.load(f)
            with open(INDEX_DIR / "chunks_code.pkl", "rb") as f:
                chunks_code = pickle.load(f)
            return bm25_docs, chunks_docs, bm25_code, chunks_code

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Index not found in {INDEX_DIR}. Please run 'index' first."
        )


def search_bm25(
    bm25: BM25Okapi,
    chunks: List[Chunk],
    query: str,
    k: int = 10
) -> List[Chunk]:
    """Return the top-k chunks from a BM25 index for the given query.

    Only chunks with a BM25 score strictly greater than zero are included;
    if fewer than ``k`` chunks have a positive score, a shorter list is
    returned.

    Args:
        bm25: A fitted ``BM25Okapi`` instance to query.
        chunks: The list of ``Chunk`` objects that was used to build ``bm25``;
            indices must correspond 1-to-1.
        query: Natural-language or code search query.
        k: Maximum number of results to return (default: 10).

    Returns:
        A list of up to ``k`` ``Chunk`` objects ranked by descending BM25
        score, excluding zero-score results.
    """
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)
    score_index_pairs = [(scores[i], i) for i in range(len(scores))]
    score_index_pairs.sort(reverse=True, key=lambda x: x[0])
    top_indices = [
        index for score, index in score_index_pairs[:k] if score > 0
    ]
    return [chunks[i] for i in top_indices]
