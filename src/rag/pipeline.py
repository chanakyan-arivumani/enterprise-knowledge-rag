import psycopg
from rag.embeddings import generate_embeddings, EMBEDDING_MODEL
from rag.models import Citation
from rag.retrieval import retrieve_chunks_by_vector
from rag.generation import (
    build_cited_context,
    generate_answer,
    validate_citation_labels,
    build_citations,
    REFUSAL_ANSWER,
    CitationValidationError,
)


def answer_question(
    conn: psycopg.Connection,
    query: str,
    k: int = 3,
) -> tuple[str, list[Citation]]:
    if not query.strip():
        raise ValueError("Query can't be empty")

    if k <= 0:
        raise ValueError("k must be greater than zero")
    query_embedding = generate_embeddings(query, EMBEDDING_MODEL)
    results = retrieve_chunks_by_vector(
        conn,
        query_embedding,
        embedding_model=EMBEDDING_MODEL,
        k=k,
    )
    if not results:
        return REFUSAL_ANSWER, []
    context, citation_map = build_cited_context(results)
    answer = generate_answer(query, context)

    cited_labels = validate_citation_labels(answer, citation_map)

    if not cited_labels and answer.strip() != REFUSAL_ANSWER:
        raise CitationValidationError("Generated answer is missing citations")

    citations = build_citations(results, citation_map, cited_labels)
    return answer, citations
