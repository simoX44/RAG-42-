from typing import List, Any, Sequence
from pydantic import BaseModel, Field, AliasChoices
import uuid


def is_valid_inp(query: Any, k: int) -> bool:
    """Check whether a query string and result count are valid inputs.

    Args:
        query: The search query to validate; must be a non-empty string.
        k: The number of results to retrieve; must be a positive integer.

    Returns:
        ``True`` if ``query`` is a non-empty string and ``k`` is a positive
        integer; ``False`` otherwise.
    """
    if not isinstance(query, str):
        return False
    if not query.strip():
        return False
    if not isinstance(k, int) or k <= 0:
        return False
    return True


class MinimalSource(BaseModel):
    """A reference to a character span within a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A dataset question that has not yet been answered."""

    # This AliasChoices grabs the real ID from the file if it's named "id"
    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        validation_alias=AliasChoices('question_id', 'id')
    )
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A dataset question together with its ground-truth sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A collection of answered and unanswered RAG evaluation questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for a single question, without a generated answer."""

    question_id: str
    question_str: str  # Corrected for Moulinette
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources for a single question, with a generated answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Per-question retrieval results produced by a student system."""

    search_results: Sequence[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """``StudentSearchResults`` extended with a generated answer
    per question."""

    search_results: List[MinimalAnswer]
