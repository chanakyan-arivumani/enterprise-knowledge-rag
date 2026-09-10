import pytest
from rag.generation import (
    calculate_generation_metrics,
    extract_citations,
    evaluate_citations,
    evaluate_generation_result,
)
from tests.mock.mock_generation_result import (
    mixed_result,
    all_answerable_result,
    all_unanswerable_result,
)


def test_calculate_generation_metrics_all_answerable():
    metrics = calculate_generation_metrics(all_answerable_result)
    assert metrics == {
        "answer_correctness": 1.0,
        "citation_correctness_if_present": 1.0,
        "citation_presence": 1.0,
        "groundedness": 1.0,
        "refusal_correctness": None,
    }


def test_calculate_generation_metrics_all_unanswerable():
    metrics = calculate_generation_metrics(all_unanswerable_result)
    assert metrics == {
        "answer_correctness": None,
        "citation_correctness_if_present": None,
        "citation_presence": None,
        "groundedness": None,
        "refusal_correctness": 0.6666666666666666,
    }


def test_calculate_generation_metrics_mixed():
    metrics = calculate_generation_metrics(mixed_result)
    assert metrics == {
        "answer_correctness": 0.75,
        "citation_correctness_if_present": 0.6666666666666666,
        "citation_presence": 0.75,
        "groundedness": 0.75,
        "refusal_correctness": 0.6666666666666666,
    }


def test_extract_citations():
    mock = {
        "Answer [Chunk 4]": {4},
        "Answer. Citation: Chunk 3": {3},
        "Answer: Chunk 3 and Chunk 5": {3, 5},
        "Answer with no citation": set(),
        "Answer [Chunk 3], see Chunk 3 again": {3},
        "According to Chunk 3, the population was around 8.5 million. Citation: Chunk 1": {
            3,
            1,
        },
        " Bengaluru is at an altitude of 900 m. [Chunk 4]": {4},
    }
    for answer, expected_chunk_ids in mock.items():
        assert expected_chunk_ids == extract_citations(answer)


def test_evaluate_citations():
    expected = {2, 3}
    extracted = {3}
    assert evaluate_citations(expected, extracted)

    expected = {3}
    extracted = {3, 7}
    assert evaluate_citations(expected, extracted)

    expected = {3}
    extracted = {7}
    assert not evaluate_citations(expected, extracted)

    expected = {3}
    extracted = set()
    assert not evaluate_citations(expected, extracted)


def test_evaluate_generation_result():
    evaluation_item = {
        "question": "This is a test question",
        "expected_chunks": set(),
        "expected_answer": set(),
        "retrieved_chunks": set(),
        "answerable": True,
        "generated_answer": "",
        "citation_present": False,
        "citation_correct": False,
    }
    evaluation_item["expected_chunks"] = {4}
    generated_answer = "... [Chunk 4]"
    result = evaluate_generation_result(evaluation_item, generated_answer)
    assert result["citation_present"] == True
    assert result["citation_correct"] == True

    evaluation_item["expected_chunks"] = {4}
    generated_answer = "Bengaluru is at an altitude of 900m."
    result = evaluate_generation_result(evaluation_item, generated_answer)
    assert result["citation_present"] == False
    assert result["citation_correct"] == False

    evaluation_item["expected_chunks"] = {4}
    generated_answer = "... [Chunk 3]"
    result = evaluate_generation_result(evaluation_item, generated_answer)
    assert result["citation_present"] == True
    assert result["citation_correct"] == False

    evaluation_item["answerable"] = False
    generated_answer = "I don't know based on the provided context."
    result = evaluate_generation_result(evaluation_item, generated_answer)
    assert result["citation_present"] == False
    assert result["citation_correct"] == None
