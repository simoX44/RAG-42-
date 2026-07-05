import fire
import json
from pathlib import Path
from tqdm import tqdm
import sys

from src.chunking import retrieve_files
from src.indexing import build_and_save_index, load_index, search_bm25
from src.models import (
    RagDataset, MinimalSearchResults, MinimalSource,
    StudentSearchResults, MinimalAnswer, StudentSearchResultsAndAnswer,
    is_valid_inp
)
from src.evaluation import evaluate_results
from typing import Any


class RAGPipeline:
    """CLI interface for the RAG pipeline: index, search, evaluate, answer."""

    PATH_CACHE = Path("data/processed/answer_cache.json")

    def index(
        self,
        max_chunk_size: int = 2000,
        repo_root: str = "data/raw/vllm-0.10.1",
    ) -> None:
        """Index the repository files into BM25-searchable chunk units.

        Args:
            max_chunk_size: Maximum characters per chunk (default: 2000).
            repo_root: Path to the repository to index.
        """
        if is_valid_inp("valid", max_chunk_size) is False:
            print(
                "Error: max_chunk_size must be a positive integer.",
                file=sys.stderr,
            )
            return
        try:
            print(
                f"Starting indexing for {repo_root}"
                f" with chunk size {max_chunk_size}..."
            )
            chunks: Any = retrieve_files(repo_root, max_chunk_size)
            if not chunks:
                print(
                    "Error: No files found or chunked."
                    " Please check the repo path."
                )
                return
            build_and_save_index(chunks)
            print("Ingestion complete! Indices saved under data/processed/")
        except Exception as e:
            print(
                f"An error occurred during indexing: {e}",
                file=sys.stderr,
            )

    def search(self, query: str, k: int = 10) -> None:
        """Search both indexes for a single query and print the top-k results.

        Args:
            query: Natural-language or code search query.
            k: Number of results to return (default: 10).
        """
        if is_valid_inp(query, k) is False:
            print(
                "Error: Invalid input. Query must be a non-empty string"
                " and k must be a positive integer.",
                file=sys.stderr,
            )
            return

        try:
            if len(query.strip()) == 0:
                print(
                    "Error: Query string is empty."
                    " Please provide a valid question.",
                    file=sys.stderr,
                )
                return
            bm25_docs, chunks_docs, bm25_code, chunks_code = load_index(
                "all"
            )
            docs_results = search_bm25(bm25_docs, chunks_docs, query, k)
            code_results = search_bm25(bm25_code, chunks_code, query, k)
            results = (docs_results + code_results)[:k]

            print(f"\nTop results for: '{query}'\n" + "=" * 40)
            for i, chunk in enumerate(results):
                print(
                    f"{i + 1}. {chunk.file_path}"
                    f" (Chars {chunk.start_char}-{chunk.end_char})"
                )
                print(f"Snippet: {chunk.text[:100]}...\n")
        except FileNotFoundError as e:
            print(f"Index load failure: {e}", file=sys.stderr)
        except Exception as e:
            print(
                f"An unexpected error occurred during search: {e}",
                file=sys.stderr,
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = 10,
        save_directory: str = "data/output/search_results",
    ) -> None:
        """Process multiple batch questions from a JSON dataset.

        Args:
            dataset_path: Path to a JSON file containing a ``RagDataset``.
            k: Number of results per question (default: 10).
            save_directory: Directory where the output JSON is written.
        """
        if is_valid_inp("valid", k) is False:
            print(
                "Error: k must be a positive integer.",
                file=sys.stderr,
            )
            return
        try:
            if "docs" in dataset_path:
                index_type = "docs"
            elif "code" in dataset_path:
                index_type = "code"
            else:
                index_type = "docs"

            bm25, chunks = load_index(index_type)
            print(f"Using {index_type} index ({len(chunks)} chunks)")

            with open(dataset_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            dataset = RagDataset.model_validate(raw_data)
            search_results = []
            print(
                f"Processing {len(dataset.rag_questions)} questions..."
            )

            query_cache: dict[str, Any] = {}
            cache_hits = 0
            for question in tqdm(
                dataset.rag_questions, desc="Searching queries"
            ):
                if question.question in query_cache:
                    retrieved = query_cache[question.question]
                    cache_hits += 1
                else:
                    retrieved = search_bm25(
                        bm25, chunks, question.question, k
                    )
                    query_cache[question.question] = retrieved

                sources = [
                    MinimalSource(
                        file_path=c.file_path,
                        first_character_index=c.start_char,
                        last_character_index=c.end_char,
                    )
                    for c in retrieved
                ]

                search_results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question_str=question.question,
                        retrieved_sources=sources,
                    )
                )

            final_output = StudentSearchResults(
                search_results=search_results, k=k
            )

            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / Path(dataset_path).name

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_output.model_dump_json(indent=2))

            print(f"Saved student_search_results to {save_path}")
            print(
                f"Cache: bypassed {cache_hits} redundant searches."
            )
        except FileNotFoundError as e:
            print(f"File lookup failure: {e}", file=sys.stderr)
        except Exception as e:
            print(
                "An unexpected error occurred during dataset"
                f" batch search: {e}",
                file=sys.stderr,
            )

    def evaluate(
        self,
        student_answer_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Evaluate search results against ground-truth sources.

        Args:
            student_answer_path: Path to the student ``StudentSearchResults``
                JSON file.
            dataset_path: Path to the ground-truth ``RagDataset`` JSON file.
            k: Maximum K value to report recall for (default: 10).
            max_context_length: Unused; kept for CLI compatibility.
        """
        try:
            evaluate_results(student_answer_path, dataset_path, k)
        except Exception as e:
            print(
                f"An error occurred during evaluation parsing: {e}",
                file=sys.stderr,
            )

    def answer(self, query: str, k: int = 10) -> None:
        """Answer a single question using retrieved context from both indexes.

        Args:
            query: Natural-language question to answer.
            k: Number of context chunks to retrieve (default: 10).
        """
        if is_valid_inp(query, k) is False:
            print(
                "Error: Invalid input. Query must be a non-empty string"
                " and k must be a positive integer.",
                file=sys.stderr,
            )
            return
        try:
            if len(query.strip()) == 0:
                print(
                    "Error: Query string is empty."
                    " Please provide a valid question.",
                    file=sys.stderr,
                )
                return
            # Lazy import — avoids loading torch on index/search commands.
            from src.generator import RAGGenerator
            bm25_docs, chunks_docs, bm25_code, chunks_code = load_index(
                "all"
            )
            print("Searching for context...")

            docs_results = search_bm25(bm25_docs, chunks_docs, query, k)
            code_results = search_bm25(bm25_code, chunks_code, query, k)
            results = (docs_results + code_results)[:k]

            generator = RAGGenerator()
            print("\nGenerating answer...\n" + "=" * 40)
            print(f"Retrieved {len(results)} chunks.")
            for i, c in enumerate(results):
                print(f"Chunk {i}: {c.file_path}")
            answer_text = generator.generate_answer(query, results)
            print(answer_text)
        except FileNotFoundError as e:
            print(f"Index load failure: {e}", file=sys.stderr)
        except Exception as e:
            print(
                "An unexpected error occurred during answer"
                f" generation: {e}",
                file=sys.stderr,
            )

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a batch of previously retrieved search results.

        Answers are cached to disk after each question so the command is safe
        to interrupt and resume.

        Args:
            student_search_results_path: Path to a ``StudentSearchResults``
                JSON file produced by ``search_dataset``.
            save_directory: Directory where the output JSON is written.
        """
        try:
            from src.generator import RAGGenerator

            with open(
                student_search_results_path, "r", encoding="utf-8"
            ) as f:
                search_data = StudentSearchResults.model_validate_json(
                    f.read()
                )

            # Load disk cache.
            self.PATH_CACHE.parent.mkdir(parents=True, exist_ok=True)
            if self.PATH_CACHE.exists():
                with open(self.PATH_CACHE, "r", encoding="utf-8") as f:
                    answer_cache: dict[str, str] = json.load(f)
                print(
                    f"Loaded {len(answer_cache)} cached answers"
                    f" from {self.PATH_CACHE}"
                )
            else:
                answer_cache = {}

            # Route to the correct index.
            if "docs" in student_search_results_path:
                index_type = "docs"
            elif "code" in student_search_results_path:
                index_type = "code"
            else:
                index_type = "docs"

            _, chunks = load_index(index_type)

            # Build chunk lookup map.
            chunk_map = {
                (c.file_path, c.start_char, c.end_char): c
                for c in chunks
            }

            generator = RAGGenerator()
            answers = []
            print(
                f"Loaded {len(search_data.search_results)} questions"
                f" from {student_search_results_path}"
            )

            for result in tqdm(
                search_data.search_results, desc="Generating Answers"
            ):
                # Reconstruct Chunk objects from saved MinimalSource entries.
                retrieved_chunks = [
                    chunk_map[key]
                    for c in result.retrieved_sources
                    if (
                        key := (
                            c.file_path,
                            c.first_character_index,
                            c.last_character_index,
                        )
                    ) in chunk_map
                ]

                # Use cache or generate.
                if result.question_str in answer_cache:
                    answer_text = answer_cache[result.question_str]
                else:
                    answer_text = generator.generate_answer(
                        result.question_str, retrieved_chunks
                    )
                    answer_cache[result.question_str] = answer_text
                    with open(
                        self.PATH_CACHE, "w", encoding="utf-8"
                    ) as f:
                        json.dump(answer_cache, f, indent=2)

                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question_str=result.question_str,
                        retrieved_sources=result.retrieved_sources,
                        answer=answer_text,
                    )
                )

            final_output = StudentSearchResultsAndAnswer(
                search_results=answers,
                k=search_data.k,
            )

            save_dir = Path(save_directory)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / Path(student_search_results_path).name

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(final_output.model_dump_json(indent=2))

            print(
                f"Processed {len(answers)}"
                f" of {len(search_data.search_results)} questions"
            )
            print(
                f"Saved student_search_results_and_answer to {save_path}"
            )
        except FileNotFoundError as e:
            print(
                f"File dependency trace missing: {e}", file=sys.stderr
            )
        except Exception as e:
            print(
                f"Critical error during batch inference loop: {e}",
                file=sys.stderr,
            )


if __name__ == '__main__':
    fire.Fire(RAGPipeline)
