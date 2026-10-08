import ollama
import re
from textwrap import dedent
from rag.retrieval import RetrievalResult
from rag.models import Citation

LARGE_MODEL = "qwen3.5:9b"
SMALL_MODEL = "qwen3.5:0.8b"
REFUSAL_ANSWER = "I don't know based on the provided context."


def generate_answer(query: str, context: str) -> str:
    prompt = dedent(
        f"""
    Answer the question directly using information from the provided context.
    Pay particular attention to what the question is asking for.

    Use only the provided context.

    Apply the first matching rule:

    1. If relevant chunks give conflicting answers and the context does not
       establish which statement applies, explain the disagreement.
       Cite each conflicting statement using (Chunk N).
       State that the applicable answer cannot be determined from the context.

    2. If the context supports an answer, answer directly.
       Cite the supporting chunks using (Chunk N).

    3. If the requested information is missing from the context, respond exactly:
       I don't know based on the provided context.
       Include no citations in this response.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """
    )

    result = ollama.generate(
        model=LARGE_MODEL,
        prompt=prompt,
        think=False,
        options={"temperature": 0},
    )
    return result["response"]


def safe_divide(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return numerator / denominator


def calculate_generation_metrics(
    results: list[dict],
) -> dict[str, float | None]:
    correct_answers = 0
    total_answerable_questions = 0
    answers_with_citations = 0
    grounded_answers = 0
    correct_citations = 0
    total_unanswerable_questions = 0
    correct_refusals = 0

    for result in results:
        if result["answerable"]:
            total_answerable_questions += 1
            if result["answer_correct"]:
                correct_answers += 1
            if result["grounded"]:
                grounded_answers += 1
            if result["citation_present"]:
                answers_with_citations += 1
                if result["citation_correct"]:
                    correct_citations += 1
        else:
            total_unanswerable_questions += 1
            if result["answer_correct"]:
                correct_refusals += 1

    return {
        "answer_correctness": safe_divide(
            correct_answers,
            total_answerable_questions,
        ),
        "groundedness": safe_divide(
            grounded_answers,
            total_answerable_questions,
        ),
        "citation_presence": safe_divide(
            answers_with_citations,
            total_answerable_questions,
        ),
        "citation_correctness_if_present": safe_divide(
            correct_citations,
            answers_with_citations,
        ),
        "refusal_correctness": safe_divide(
            correct_refusals,
            total_unanswerable_questions,
        ),
    }


def run_generation_evaluation(
    evaluation_dataset: list[dict], chunks: list[dict], k: int
) -> list[dict]:
    results = []
    from cosine_similarity import retrieve, build_context

    for evaluation_item in evaluation_dataset:
        retrieved = retrieve(evaluation_item["question"], chunks, k)
        context = build_context(retrieved)

        generated_answer = generate_answer(evaluation_item["question"], context)
        evaluation_result = evaluate_generation_result(
            evaluation_item, generated_answer
        )
        evaluation_result["retrieved_chunks"] = [r.chunk_id for r in retrieved]
        results.append(evaluation_result)

    return results


def extract_citations(answer: str) -> set[int]:
    return {int(chunk_id) for chunk_id in re.findall(r"Chunk (\d+)", answer)}


def evaluate_citations(
    expected_chunk_ids: set[int], extracted_chunk_ids: set[int]
) -> bool:
    return bool(expected_chunk_ids & extracted_chunk_ids)


def evaluate_generation_result(evaluation_item: dict, generated_answer: str) -> dict:
    citations = set()
    citation_correct = None
    if evaluation_item["answerable"]:
        citations = extract_citations(generated_answer)
        citation_correct = evaluate_citations(
            evaluation_item["expected_chunks"], citations
        )
    return {
        "question": evaluation_item["question"],
        "expected_chunks": evaluation_item["expected_chunks"],
        "retrieved_chunks": evaluation_item["retrieved_chunks"],
        "answerable": evaluation_item["answerable"],
        "expected_answer": evaluation_item["expected_answer"],
        "generated_answer": generated_answer,
        "citation_present": bool(citations),
        "citation_correct": citation_correct,
        "answer_correct": True,
        "grounded": True,
    }


def build_cited_context(
    results: list[RetrievalResult],
    *,
    max_context_chars: int,
) -> tuple[str, dict[int, str]]:
    """Return evidence text and a citation-label-to-chunk-ID mapping.
    Chunk 1
    The annual learning allowance is INR 30000 per employee.

    Chunk 2
    Employees can work remotely up to three days per week.

    {
    1: "stored-chunk-id-for-learning-allowance",
    2: "stored-chunk-id-for-remote-work",
    }
    """
    if max_context_chars <= 0:
        raise ValueError("max_context_chars must be greater than zero")
    accepted_blocks = []
    char_count = 0
    citation_map = {}
    idx = 1

    for item in results:
        block = f"Chunk {idx}\n{item.text}"
        additional_chars = len(block) + (2 if accepted_blocks else 0)
        if char_count + additional_chars <= max_context_chars:
            accepted_blocks.append(block)
            citation_map[idx] = item.chunk_id
            char_count += additional_chars
            idx += 1
    context = "\n\n".join(accepted_blocks)
    return context, citation_map


class CitationValidationError(RuntimeError):
    """The generated answer contains invalid citation references."""


class ContextBudgetError(RuntimeError):
    """No retrieved evidence fits within the configured context budget."""


def validate_citation_labels(
    answer: str,
    citation_map: dict[int, str],
) -> set[int]:
    citations_present = extract_citations(answer)
    available_citations = set(citation_map.keys())
    invalid_citations = citations_present - available_citations
    if invalid_citations:
        raise CitationValidationError(
            f"Unknown citation labels: {sorted(invalid_citations)}"
        )
    return citations_present


def build_citations(
    results: list[RetrievalResult],
    citation_map: dict[int, str],
    cited_labels: set[int],
) -> list[Citation]:
    chunk_id_to_result = {result.chunk_id: result for result in results}
    citations = []
    for cited_label in sorted(cited_labels):
        chunk_id = citation_map[cited_label]
        result = chunk_id_to_result[chunk_id]
        citation = Citation(
            label=cited_label,
            chunk_id=chunk_id,
            document_id=result.document_id,
            source=result.source,
            excerpt=result.text,
        )
        citations.append(citation)
    return citations
