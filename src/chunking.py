from dataclasses import dataclass
from typing import List, Tuple, Set
from pathlib import Path
import ast


@dataclass
class Chunk:
    file_path: str
    start_char: int
    end_char: int
    text: str
    kind: str


def read_file(file: Path) -> str:
    """Read file content with consistent line endings.

    Args:
        file: Path to the file to read.

    Returns:
        The file contents as a string, or an empty string if the file
        cannot be decoded or does not exist.
    """
    try:
        with open(file, "r", encoding="utf-8", newline="") as f:
            return f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return ""


def find_smart_split_point(
    text: str, start: int, max_size: int, is_python: bool
) -> int:
    """Find a natural split point within max_size characters from start.

    Searches backwards from ``start + max_size`` for a language-appropriate
    boundary (class/function boundaries for Python, heading/paragraph breaks
    for Markdown).  Falls back to the hard limit if no boundary is found.

    Args:
        text: The full source text.
        start: The character offset at which the current chunk begins.
        max_size: Maximum number of characters to include from ``start``.
        is_python: ``True`` to use Python breakpoints; ``False`` for Markdown.

    Returns:
        A character offset within ``text`` that marks a natural end position,
        at most ``start + max_size`` characters from the beginning.
    """
    if start + max_size >= len(text):
        return len(text)

    window = text[start:start + max_size]
    breakpoints = ["\nclass ", "\ndef ", "\n\n", "\n"] if is_python \
        else ["\n# ", "\n## ", "\n\n", "\n"]

    for bp in breakpoints:
        idx = window.rfind(bp)
        if idx > 0:
            return start + idx

    return start + max_size


def get_char_offset(text: str, line_number: int) -> int:
    """Convert a zero-based line number to a character offset in text.

    Args:
        text: The source text to index into.
        line_number: Zero-based line number to convert.

    Returns:
        The character offset of the start of ``line_number``, clamped to
        ``len(text)`` if the line number exceeds the text length.
    """
    lines = text.split("\n")
    offset = 0
    for i, line in enumerate(lines):
        if i >= line_number:
            break
        offset += len(line) + 1
    return min(offset, len(text))


def split_and_make_chunks(
    path_str: str,
    raw_text: str,
    base_start: int,
    context_header: str,
    max_chunk_size: int
) -> List[Chunk]:
    """Split raw_text into chunks with context_header prepended to each.

    Character offsets stored on each ``Chunk`` are relative to the original
    file (``base_start`` is added to all local positions).

    Args:
        path_str: File path stored on every produced ``Chunk``.
        raw_text: The text to split (not including the header).
        base_start: Character offset of ``raw_text`` within the original file.
        context_header: Header string prepended to each chunk's text.
        max_chunk_size: Maximum total characters (header + body) per chunk.

    Returns:
        A list of ``Chunk`` objects, possibly empty if ``raw_text`` is blank.
    """
    chunks: List[Chunk] = []
    effective_limit = max_chunk_size - len(context_header)
    start = 0

    while start < len(raw_text):
        end = find_smart_split_point(
            raw_text, start, effective_limit, is_python=True
        )
        sub_text = raw_text[start:end].strip()
        if sub_text:
            chunks.append(Chunk(
                file_path=path_str,
                start_char=base_start + start,
                end_char=base_start + end,
                text=context_header + sub_text,
                kind=".py"
            ))
        start = end

    return chunks


def get_uncovered_ranges(
    text_len: int,
    covered_ranges: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Return ranges in ``[0, text_len)`` not covered by any AST chunk.

    Args:
        text_len: Total length of the source text in characters.
        covered_ranges: List of ``(start, end)`` pairs already claimed by AST
            chunks.  Ranges may be unsorted and need not be contiguous.

    Returns:
        A sorted list of ``(start, end)`` pairs representing uncovered gaps,
        including the leading gap before the first covered range and the
        trailing gap after the last, if present.  Returns
        ``[(0, text_len)]`` when ``covered_ranges`` is empty.
    """
    if not covered_ranges:
        return [(0, text_len)]

    covered = sorted(covered_ranges)
    uncovered = []

    if covered[0][0] > 0:
        uncovered.append((0, covered[0][0]))

    for i in range(len(covered) - 1):
        if covered[i + 1][0] > covered[i][1]:
            uncovered.append((covered[i][1], covered[i + 1][0]))

    if covered[-1][1] < text_len:
        uncovered.append((covered[-1][1], text_len))

    return uncovered


DOC_SYNONYMS = {
    "cli": "command line interface command",
    "deployment": "deploy run serve launch",
    "installation": "install setup pip",
    "quantization": "compress optimize int8 fp8",
    "kubernetes": "k8s cluster container pod",
    "benchmark": "latency throughput performance speed",
    "contributing": "contribute develop build source",
    "integration": "connect plugin framework tool",
    "configuration": "config setting parameter argument",
    "troubleshooting": "error fix debug problem issue",
}


def enrich_path_keywords(path_str: str) -> str:
    """Extract path components as keywords and expand with domain synonyms.

    Path separators, hyphens, underscores, and common extensions are replaced
    with spaces.  Known domain terms (e.g. ``"cli"``, ``"benchmark"``) are
    augmented with their synonyms from ``DOC_SYNONYMS``.

    Args:
        path_str: A file path string (relative or absolute).

    Returns:
        A lower-cased, space-separated keyword string derived from the path,
        enriched with any matching ``DOC_SYNONYMS`` expansions.
    """
    path = Path(path_str).parts
    keywords = " ".join(path).lower()
    keywords = (
        keywords
        .replace('-', ' ')
        .replace('_', ' ')
        .replace('.md', '')
        .replace('.txt', '')
    )
    for key, synonyms in DOC_SYNONYMS.items():
        if key in keywords:
            keywords += f" {synonyms}"
    return keywords


def chunk_file_by_chars(
    path_str: str,
    text: str,
    max_chunk_size: int,
    is_python: bool = False
) -> List[Chunk]:
    """Split text into character-based chunks.

    Markdown/text files use a sliding window with 100-character overlap and a
    keyword-enriched header; Python files use non-overlapping fixed splits with
    no header.

    Args:
        path_str: File path stored on every produced ``Chunk``.
        text: Full file content to split.
        max_chunk_size: Target maximum characters per chunk (may be reduced for
            Markdown files to stay within the 2000-character total limit).
        is_python: ``True`` for Python source files; ``False`` for Markdown or
            plain text.

    Returns:
        A list of ``Chunk`` objects covering the entire text.
    """
    kind = ".py" if is_python else ".md"
    chunks: List[Chunk] = []
    context_header = ""
    overlap = 0

    if kind in (".md", ".txt"):
        filename = (
            Path(path_str).stem
            .replace('-', ' ')
            .replace('_', ' ')
        )
        context_header = (
            f"# File: {path_str}\n"
            f"# Topic: {filename}\n"
            f"# Keywords: {enrich_path_keywords(path_str)}\n"
        )
        max_chunk_size = 1200
        overlap = 100
        if (max_chunk_size + len(context_header)) > 2000:
            max_chunk_size = 2000 - len(context_header)

    start = 0
    while start < len(text):
        end = find_smart_split_point(text, start, max_chunk_size, is_python)
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(
                file_path=path_str,
                start_char=start,
                end_char=end,
                text=context_header + chunk_text,
                kind=kind
            ))
        next_start = end - overlap
        start = next_start if next_start > start else end

    return chunks


def chunk_python_ast(
    path_str: str,
    text: str,
    max_chunk_size: int
) -> List[Chunk]:
    """Chunk a Python source file using AST node boundaries.

    Each top-level function, class header, and method is emitted as a separate
    chunk.  Methods include their full class hierarchy in the header.  Nested
    classes are handled recursively.  Large nodes are split further by
    character count.  Any text not covered by an AST node (imports,
    module-level statements) is collected and emitted as additional chunks.
    All character
    offsets refer to positions in the original file.

    Falls back to ``chunk_file_by_chars`` if the file cannot be parsed.

    Args:
        path_str: File path stored on every produced ``Chunk``.
        text: Full Python source text.
        max_chunk_size: Maximum characters per chunk (header included).

    Returns:
        A list of ``Chunk`` objects covering the entire file.
    """
    chunks: List[Chunk] = []

    try:
        tree = ast.parse(text)
    except SyntaxError:
        return chunk_file_by_chars(
            path_str, text, max_chunk_size, is_python=True
        )

    covered_ranges: List[Tuple[int, int]] = []

    # Recursive function to process classes and their methods
    def process_class(node: ast.ClassDef, parent_classes: List[str]) -> None:
        class_hierarchy = parent_classes + [node.name]
        class_context = " > ".join(class_hierarchy)

        # Class header chunk
        class_start = get_char_offset(text, node.lineno - 1)
        assert node.end_lineno is not None
        end_line = min(node.lineno + 5, node.end_lineno)
        class_end = get_char_offset(text, end_line)
        class_header = text[class_start:class_end].strip()

        if class_header:
            chunks.append(Chunk(
                file_path=path_str,
                start_char=class_start,
                end_char=class_end,
                text=(
                    f"# File: {path_str}\n"
                    f"# Class: {class_context}\n"
                    f"{class_header}"
                ),
                kind=".py"
            ))

        # Process everything inside this class
        for item in node.body:

            if isinstance(item, ast.ClassDef):
                # Nested class — recurse with updated hierarchy
                process_class(item, class_hierarchy)  # recursive call

            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Method — chunk with full class context
                start_char = get_char_offset(text, item.lineno - 1)
                assert item.end_lineno is not None
                end_char = get_char_offset(text, item.end_lineno)
                method_text = text[start_char:end_char].strip()

                if method_text:
                    # Add class context to header
                    header = (
                        f"# File: {path_str}\n"
                        f"# Class: {class_context}\n"
                        f"# Method: {item.name}\n"
                    )
                    chunks.extend(split_and_make_chunks(
                        path_str, method_text,
                        start_char, header, max_chunk_size
                    ))
                    covered_ranges.append((start_char, end_char))

    # Before main loop — run ONCE
    method_nodes: Set[int] = set()
    nested_classes: Set[int] = set()

    for parent in ast.walk(tree):
        if isinstance(parent, ast.ClassDef):
            for item in parent.body:
                # Track methods
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_nodes.add(id(item))
                # Track nested classes
                if isinstance(item, ast.ClassDef):
                    nested_classes.add(id(item))

    for node in ast.walk(tree):
        # Classes
        if isinstance(node, ast.ClassDef):
            # Skip nested classes — they are handled in process_class
            if id(node) in nested_classes:
                continue
            process_class(node, [])

        # Top-level functions
        elif isinstance(node, ast.FunctionDef):
            # Skip methods — they are handled inside ClassDef above
            if id(node) in method_nodes:
                continue

            start_char = get_char_offset(text, node.lineno - 1)
            assert node.end_lineno is not None
            end_char = get_char_offset(text, node.end_lineno)
            func_text = text[start_char:end_char].strip()

            if func_text:
                header = f"# File: {path_str}\n# Function: {node.name}\n"
                chunks.extend(split_and_make_chunks(
                    path_str, func_text, start_char, header, max_chunk_size
                ))
                covered_ranges.append((start_char, end_char))

    # Uncovered text (imports, module-level code)
    for unc_start, unc_end in get_uncovered_ranges(len(text), covered_ranges):
        snippet = text[unc_start:unc_end].strip()
        if snippet:
            header = f"# File: {path_str}\n"
            chunks.extend(split_and_make_chunks(
                path_str, snippet, unc_start, header, max_chunk_size
            ))

    return chunks


def chunk_file(file: Path, path_str: str, chunk_size: int) -> List[Chunk]:
    """Dispatch to the appropriate chunking strategy based on file extension.

    ``.py`` files use AST-aware chunking; all other supported types use
    character-based chunking.  Returns an empty list for empty files.

    Args:
        file: ``Path`` object used to read the file and inspect its suffix.
        path_str: File path string stored on every produced ``Chunk``.
        chunk_size: Maximum characters per chunk passed to the chunker.

    Returns:
        A list of ``Chunk`` objects, or an empty list if the file is empty.
    """
    text = read_file(file)
    if not text.strip():
        return []

    if file.suffix == ".py":
        return chunk_python_ast(path_str, text, chunk_size)
    else:
        return chunk_file_by_chars(path_str, text, chunk_size, is_python=False)


def retrieve_files(
    repo_root: str,
    max_chunk_size: int = 2000
) -> List[Chunk]:
    """Recursively find and chunk all valid files under ``repo_root``.

    Only files with extensions ``.py``, ``.md``, or ``.txt`` are processed.

    Args:
        repo_root: Root directory to search recursively.
        max_chunk_size: Maximum characters per chunk (default: 2000).

    Returns:
        A flat list of ``Chunk`` objects from all discovered files.
    """
    chunks: List[Chunk] = []
    repo_path = Path(repo_root)
    valid_extensions = {".py", ".md", ".txt"}

    for file in repo_path.rglob("*"):
        if file.is_file() and file.suffix in valid_extensions:
            path_str = str(file)
            chunks.extend(chunk_file(file, path_str, max_chunk_size))

    return chunks
