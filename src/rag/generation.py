import ollama
import re
from textwrap import dedent

LARGE_MODEL = "qwen3.5:9b"
SMALL_MODEL = "qwen3.5:0.8b"


def generate_answer(query: str, context: str) -> str:
    prompt = dedent(
        f"""
    Answer the question directly using information from the provided context.
    Pay particular attention to what the question is asking for.

    Use only the provided context.

    If the answer is not present in the context, respond exactly:
    I don't know based on the provided context.

    If the answer is present, cite the chunk that contains the answer.

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
