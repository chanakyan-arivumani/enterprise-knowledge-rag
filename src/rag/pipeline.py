import psycopg
from rag.embeddings import generate_embeddings, EMBEDDING_MODEL
from rag.retrieval import retrieve_chunks_by_vector
from rag.generation import build_cited_context, generate_answer


def answer_question(
    conn: psycopg.Connection,
    query: str,
    k: int = 3,
) -> tuple[str, dict[int, str]]:
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
        return ("I don't know based on the provided context.", {})
    context, citation_map = build_cited_context(results)
    answer = generate_answer(query, context)
    return answer, citation_map
