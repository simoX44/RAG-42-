from src.models import RagDataset, StudentSearchResults


def calculate_overlap_percentage(
    s_truth: int, e_truth: int, s_retrieve: int, e_retrieve: int
) -> float:
    """Calculate the fraction of the ground-truth span covered by a retrieved
    chunk.

    Args:
        s_truth: Start character index of the ground-truth source.
        e_truth: End character index of the ground-truth source.
        s_retrieve: Start character index of the retrieved chunk.
        e_retrieve: End character index of the retrieved chunk.

    Returns:
        A float in ``[0.0, 1.0]``: the fraction of the ground-truth span
        that overlaps with the retrieved chunk.  ``0.0`` when there is no
        overlap.
    """
    overlap_start = max(s_truth, s_retrieve)
    overlap_end = min(e_truth, e_retrieve)
    overlap_size = max(0, overlap_end - overlap_start)

    truth_size = max(1, e_truth - s_truth)  # Avoid division by zero

    return overlap_size / truth_size


def evaluate_results(
    student_answer_path: str, dataset_path: str, k: int = 10
) -> None:
    """Evaluate retrieval recall of student results against ground-truth data.

    Loads student search results and ground-truth answers from JSON files,
    matches questions by ``question_id``, and computes Recall@K for
    ``K`` in ``{1, 3, 5, 10}``.  Only K values up to the supplied ``k``
    argument are printed.  Overlap between retrieved and ground-truth spans
    must be at least 5 % of the ground-truth span to count as a hit.

    Args:
        student_answer_path: Path to a JSON file containing a
            ``StudentSearchResults`` object.
        dataset_path: Path to a JSON file containing a ``RagDataset`` object
            with ground-truth ``AnsweredQuestion`` entries.
        k: Maximum K value to report (default: 10).

    Returns:
        None.  Results are printed to stdout.
    """
    try:
        with open(student_answer_path, "r") as f:
            student_data = StudentSearchResults.model_validate_json(f.read())

        with open(dataset_path, "r") as f:
            truth_data = RagDataset.model_validate_json(f.read())

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # FIX 1: Map by question_id, NOT the randomly generated UUID key.
    truth_map = {
        q.question_id: q
        for q in truth_data.rag_questions
        if hasattr(q, 'sources')
    }

    total_questions = len(truth_map)
    if total_questions == 0:
        print("No answered questions found in dataset.")
        return

    print("Student data is valid: True")
    print(f"Total number of questions: {len(truth_data.rag_questions)}")
    print(f"Total number of questions with sources: {total_questions}")
    n_student = len(student_data.search_results)
    print(f"Total number of questions with student sources: {n_student}")

    # Calculate recall at different K values
    k_values = [1, 3, 5, 10]
    recalls = {kval: 0.0 for kval in k_values}

    matched_qs = 0
    for result in student_data.search_results:
        # Check against the string to bypass UUID mismatches
        if result.question_id not in truth_map:
            continue

        matched_qs += 1
        truth_question = truth_map[result.question_id]
        truth_sources = truth_question.sources

        for kval in k_values:
            top_k_retrieved = result.retrieved_sources[:kval]
            found_count = 0

            for gt in truth_sources:
                found = False
                for retrieved in top_k_retrieved:
                    if (
                        gt.file_path.endswith(retrieved.file_path)
                        or retrieved.file_path.endswith(gt.file_path)
                    ):
                        overlap = calculate_overlap_percentage(
                            gt.first_character_index,
                            gt.last_character_index,
                            retrieved.first_character_index,
                            retrieved.last_character_index,
                        )
                        if overlap >= 0.05:  # 5% minimum overlap rule
                            found = True
                            break
                if found:
                    found_count += 1

            score = found_count / len(truth_sources)
            recalls[kval] += score

    print("\nEvaluation Results")
    print("========================================")
    print(f"Questions evaluated: {matched_qs} / {total_questions}")
    for kval in k_values:
        if kval <= k:
            avg_recall = recalls[kval] / matched_qs if matched_qs > 0 else 0.0
            print(f"Recall@{kval}: {avg_recall * 100}")
