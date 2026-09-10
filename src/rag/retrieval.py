import re
import math
import ollama
from copy import copy
from dataclasses import dataclass

# from embeddings import bengaluru_text_list, bengaluru_eval_ds


# text_list = ["I'm just a poor boy", "Nobody loves me"]
# chunks = convert_text_list_to_embed_list(bengaluru_text_list)


@dataclass
class RetrievalResult:
    chunk_id: int
    text: str
    score: float


def retrieve(query, chunks, k=3) -> list[RetrievalResult]:
    """
    chunks = [{"text": "I'm just a poor boy", "embedding": [0.12, 0.34, 0.56...]}]
    """
    query_embed = generate_embeddings(query)
    results = []
    for chunk in chunks:
        sim = get_cosine_similarity(query_embed, chunk["embedding"])
        results.append(RetrievalResult(chunk["id"], chunk["text"], sim))
    top_k = sorted(results, key=lambda x: x.score, reverse=True)[:k]
    return top_k


# ________________________________________________________________________________________
# Keyword retrieval
# ________________________________________________________________________________________


def retrieve_keyword(
    query: str,
    chunks: list[dict],
    k: int = 3,
) -> list[RetrievalResult]:
    results = []
    punctuation_removed_query = re.sub(r"[^\w\s]", "", query)
    query_tokens = set(punctuation_removed_query.lower().split())
    for chunk in chunks:
        punctuation_removed_chunk = re.sub(r"[^\w\s]", "", chunk["text"])
        chunk_tokens = set(punctuation_removed_chunk.lower().split())
        score = len(query_tokens & chunk_tokens)
        results.append(RetrievalResult(chunk["id"], chunk["text"], score))
    top_k = sorted(results, key=lambda x: x.score, reverse=True)[:k]
    return top_k


def retrieve_hybrid(
    query: str,
    chunks: list[dict],
    k: int = 3,
    alpha: float = 0.5,
) -> list[RetrievalResult]:
    """
    chunk_id | semantic_score | keyword_score | normalized_keyword
    normalized_keyword = keyword_score / max_keyword_score
    """
    keyword_results = retrieve_keyword(query, chunks, len(chunks))
    semantic_results = retrieve(query, chunks, len(chunks))
    merged = {}
    keyword_scores = []
    for k_res in keyword_results:
        merged[k_res.chunk_id] = {"keyword": k_res.score, "text": k_res.text}
        keyword_scores.append(k_res.score)
    for sem_res in semantic_results:
        merged.setdefault(sem_res.chunk_id, {})["semantic"] = sem_res.score
        merged[sem_res.chunk_id]["text"] = sem_res.text

    max_keyword_score = max(keyword_scores, default=0)

    for chunk_id, item in merged.items():
        normalized_keyword_score = 0
        keyword_score = item.get("keyword", 0)
        semantic_score = item.get("semantic", 0)
        if max_keyword_score > 0:
            normalized_keyword_score = keyword_score / max_keyword_score
        item["normalized_keyword"] = normalized_keyword_score
        item["hybrid"] = alpha * semantic_score + (1 - alpha) * normalized_keyword_score
    sorted_hybrid = sorted(
        merged.items(),
        key=lambda x: x[1]["hybrid"],
        reverse=True,
    )[:k]
    result = []
    for chunk_id, item in sorted_hybrid:
        result.append(RetrievalResult(chunk_id, item["text"], item["hybrid"]))
    return result


def build_context(results: list[RetrievalResult]):
    return "\n\n".join(f"Chunk {result.chunk_id}\n{result.text}" for result in results)


def inspect_retrieval(eval_data, chunks, k):
    """
    Query: When did humans settle in the Bengaluru region?
    Expected: 6

    Rank 1: chunk 10 | score=0.555
    Rank 2: chunk 6  | score=0.385
    Rank 3: chunk 3  | score=0.627

    Correct rank: 2
    """
    for query, expected_chunk_ids in eval_data:
        if expected_chunk_ids is None:
            continue
        print(f"Query: {query}\n")
        print(f"Expected: {expected_chunk_ids}\n")

        result_list = retrieve(query, chunks, k)
        for _id, result in enumerate(result_list):
            print(f"Rank {_id+1}: chunk {result.chunk_id} | score={result.score}\n")
            if result.chunk_id in expected_chunk_ids:
                print(f"Correct rank: {_id+1}")


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    k: int = 3,
) -> list[RetrievalResult]:
    results = []
    pairs = [(query, candidate.text) for candidate in candidates]
    # scores = reranker_model.predict(pairs)
    scores = reranker_model.compute_score(
        pairs,
        normalize=False,
    )

    for candidate, relevance in zip(candidates, scores):
        # relevance = calculate_relevance(query, candidate.text)
        new_candidate = copy(candidate)
        new_candidate.score = relevance
        results.append(new_candidate)
    return sorted(results, key=lambda x: x.score, reverse=True)[:k]
