def prepare_eval_data(eval_data, chunks):
    prepared_data = []
    for entry in eval_data:
        expected_chunks = get_expected_chunks(entry["expected_sentences"], chunks)
        prepared_data.append((entry["question"], expected_chunks))
    return prepared_data


def get_expected_chunks(expected_sentences: set[int], chunks: list[dict]) -> set[int]:
    # chunk["sentence_ids"] & expected_sentences
    expected_chunk_ids = set()
    for chunk in chunks:
        if chunk["sentence_ids"] & expected_sentences:
            expected_chunk_ids.add(chunk["id"])
    return expected_chunk_ids


def recall_at_k(eval_data, chunks, k):
    query_count_with_correct_answer = 0
    answerable_query_count = 0

    for item in eval_data:
        if not item.get("expected_sentences"):
            continue

        answerable_query_count += 1
        result_list = retrieve_hybrid(item["question"], chunks, k, alpha=0.0)
        # result_list = retrieve(item["question"], chunks, k)
        retrieved_chunk_ids = []
        for _id, result in enumerate(result_list):
            retrieved_chunk_ids.append(result.chunk_id)
            if result.chunk_id in item["expected_sentences"]:
                query_count_with_correct_answer += 1
                break
    return query_count_with_correct_answer / answerable_query_count


def mean_reciprocal_rank(eval_data, chunks, k):
    reciprocal_ranks = []
    for item in eval_data:
        if not item.get("expected_sentences"):
            continue
        # result_list = retrieve(item["question"], chunks, k)
        result_list = retrieve_hybrid(item["question"], chunks, k, alpha=0.0)
        rank = 0
        for _id, result in enumerate(result_list):
            # print(f"\nRank {_id+1}: chunk {result[1]},")
            if result.chunk_id in item["expected_sentences"]:
                rank = _id + 1
                break
        if rank:
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)

    mean_rank = sum(reciprocal_ranks) / len(reciprocal_ranks)
    return mean_rank


def calculate_retrieval_metrics(
    results: list[dict], k_values: tuple[int, ...] = (1, 2, 3, 5, 10)
) -> dict[str, float | None]:
    answerable_results = [result for result in results if result["answerable"]]

    metrics = {}

    for k in k_values:
        hits = 0

        for result in answerable_results:
            expected_chunks = result["expected_chunks"]
            retrieved_chunks = result["retrieved_chunks"][:k]
            if expected_chunks & set(retrieved_chunks):
                hits += 1
        metrics[f"recall@{k}"] = safe_divide(hits, len(answerable_results))

    for k in k_values:
        reciprocal_ranks = []
        for result in answerable_results:
            expected_chunks = result["expected_chunks"]
            retrieved_chunks = result["retrieved_chunks"][:k]
            reciprocal_rank = 0.0
            for rank, chunk_id in enumerate(retrieved_chunks, start=1):
                if chunk_id in expected_chunks:
                    reciprocal_rank = 1 / rank
                    break
            reciprocal_ranks.append(reciprocal_rank)
        metrics[f"mrr@{k}"] = safe_divide(sum(reciprocal_ranks), len(reciprocal_ranks))
    return metrics


# chunks = convert_text_list_to_embed_list(bengaluru_text_list)
def evaluate_retrieval_at_k(
    evaluation_dataset: list[dict],
    k: int,
) -> dict[str, float | None]:
    recall = recall_at_k(evaluation_dataset, chunks, k)
    mrr = mean_reciprocal_rank(evaluation_dataset, chunks, k)
    return {f"{k}": {"recall": recall, "mrr": mrr}}


def evaluate_retriever(evaluation_ds=bengaluru_eval_ds):
    chunks = convert_text_list_to_embed_list(bengaluru_text_list)
    result_list = []
    for _ in evaluation_ds:
        result = {"query": _["question"]}
        for k in (1, 3, 5):
            if not _["expected_chunks"]:
                continue
            top_k = retrieve(_["question"], chunks, k)
            recall = 0
            match = 0  # retrieved intersection relevant for the given k
            for similarity, chunk_id, text in top_k:
                if chunk_id in _["expected_chunks"]:
                    match += 1
            recall = match / len(_["expected_chunks"])
            result[f"R@{k}"] = recall
        result_list.append(result)
    return result_list


def evaluate_reranked_retrieval(
    evaluation_dataset,
    candidate_k: int = 10,
    final_k: int = 3,
):
    results = []
    for item in evaluation_dataset:
        query = item["question"]
        expected = set(item["expected_sentences"])
        candidates = retrieve_hybrid(
            query,
            chunks,
            k=candidate_k,
            alpha=0.5,
        )
        reranked = rerank(
            query,
            candidates,
            k=final_k,
        )
        retrieved = [result.chunk_id for result in reranked]
        results.append(
            {
                "question": query,
                "expected_chunks": expected,
                "retrieved_chunks": retrieved,
                "answerable": item["answerable"],
            }
        )
    return results
