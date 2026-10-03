import pytest
import pydantic
from rag.generation import (
    calculate_generation_metrics,
    extract_citations,
    evaluate_citations,
    evaluate_generation_result,
    build_cited_context,
    validate_citation_labels,
    CitationValidationError,
    build_citations,
)
from rag.models import Citation
from rag.retrieval import RetrievalResult
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


def test_cited_context_resolves_citation_to_stored_chunk():
    results = [
        RetrievalResult(
            chunk_id="chunk-z",
            text="The learning allowance is INR 30000.",
            score=0.9,
        ),
        RetrievalResult(
            chunk_id="chunk-a",
            text="Remote work is allowed three days per week.",
            score=0.8,
        ),
    ]

    context, citation_map = build_cited_context(results)

    assert "Chunk 2\nRemote work is allowed three days per week." in context

    answer = "Employees may work remotely three days per week. [Chunk 2]"
    cited_labels = extract_citations(answer)
    cited_chunk_ids = {citation_map[label] for label in cited_labels}

    assert cited_chunk_ids == {"chunk-a"}


def test_build_cited_context_empty_results():
    assert build_cited_context([]) == ("", {})


def test_validate_citation_labels():
    answer = "Chunk 1 test"
    citation_map = {1: "test1", 2: "test2", 3: "test3"}
    assert validate_citation_labels(answer, citation_map) == {1}


def test_citation_validation_error():
    answer = "Chunk 1 test\nChunk 9 test"
    citation_map = {1: "test1", 2: "test2", 3: "test3"}
    with pytest.raises(CitationValidationError):
        validate_citation_labels(answer, citation_map)


def test_validate_nil_citations():
    answer = "test"
    citation_map = {1: "test1", 2: "test2", 3: "test3"}
    assert validate_citation_labels(answer, citation_map) == set()


def test_build_citations_without_cited_labels():
    results = [
        RetrievalResult(
            chunk_id="chunk-z",
            text="The learning allowance is INR 30000.",
            score=0.9,
        ),
    ]
    citation_map = {1: "test1", 2: "test2"}
    assert build_citations(results, citation_map, set()) == []


def test_build_citations_raises_exceptions():
    results = [
        RetrievalResult(
            chunk_id="chunk-z",
            text="The learning allowance is INR 30000.",
            score=0.9,
        ),
    ]
    citation_map = {1: "chunk-z", 2: "chunk-y"}
    cited_labels = {1}
    with pytest.raises(pydantic.ValidationError):
        build_citations(results, citation_map, cited_labels)


def test_build_citations():
    results = [
        RetrievalResult(
            chunk_id="chunk-a",
            text="The learning allowance is INR 30000.",
            score=0.9,
            document_id="Document_001",
            source="/temp/learning.txt",
        ),
        RetrievalResult(
            chunk_id="chunk-b",
            text="Remote work is allowed three days per week.",
            score=0.8,
            document_id="Document_002",
            source="/temp/remote_work.txt",
        ),
        RetrievalResult(
            chunk_id="chunk-c",
            text="Travel claims must be submitted within 15 days.",
            score=0.7,
            document_id="Document_003",
            source="/temp/travel.txt",
        ),
    ]

    # Deliberately map labels differently from the results' positions.
    citation_map = {1: "chunk-c", 2: "chunk-b", 3: "chunk-a"}
    cited_labels = {3, 1}

    citations = build_citations(results, citation_map, cited_labels)

    assert citations == [
        Citation(
            label=1,
            chunk_id="chunk-c",
            document_id="Document_003",
            source="/temp/travel.txt",
            excerpt="Travel claims must be submitted within 15 days.",
        ),
        Citation(
            label=3,
            chunk_id="chunk-a",
            document_id="Document_001",
            source="/temp/learning.txt",
            excerpt="The learning allowance is INR 30000.",
        ),
    ]
