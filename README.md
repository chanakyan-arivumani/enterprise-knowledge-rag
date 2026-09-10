# Enterprise Knowledge RAG

A from-scratch Retrieval-Augmented Generation (RAG) system built to understand and evaluate the core components of an enterprise knowledge assistant.

The project focuses on understanding the RAG pipeline from first principles before introducing higher-level frameworks.

## Current Pipeline

```text
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Semantic Retrieval ──┐
                     ├── Hybrid Retrieval
Keyword Retrieval ───┘
    ↓
Context Construction
    ↓
LLM Generation
    ↓
Grounded Answer + Citations
```

## Current Implementation

* Sentence-based chunking with configurable overlap
* Embedding generation using Ollama
* Cosine similarity
* Semantic retrieval
* Keyword retrieval
* Hybrid semantic + keyword retrieval
* Context construction
* LLM answer generation
* Citation extraction and validation
* Refusal when the provided context is insufficient
* Retrieval evaluation
* Generation evaluation
* Unit tests with pytest

## Models

The current implementation uses:

* `qwen3-embedding:0.6b` for embeddings
* `qwen3.5:9b` for answer generation

Models are run locally through Ollama.

## Evaluation

The project currently includes a 35-question evaluation dataset based on a small Bengaluru knowledge corpus.

The evaluation contains:

* Answerable factual questions
* Paraphrased questions
* Multi-chunk questions
* Unanswerable questions

Retrieval is evaluated independently from generation using:

* Recall@K
* Mean Reciprocal Rank (MRR)

Generation evaluation includes:

* Answer correctness
* Groundedness
* Citation presence
* Citation correctness
* Refusal correctness

### Retrieval Results

On the current Bengaluru evaluation dataset, the hybrid retrieval baseline achieved:

* Recall@1: 1.00
* MRR@1: 1.00

The hybrid approach was compared with semantic and keyword retrieval during development.

Reranking was also experimentally evaluated. The tested reranker models did not improve Recall@1 over the hybrid baseline, so reranking is not currently part of the default retrieval pipeline.

These results are specific to the current small evaluation dataset and should not be interpreted as production-level retrieval performance.

## Project Structure

```text
src/rag/
├── __init__.py
├── chunking.py
├── embeddings.py
├── retrieval.py
├── generation.py
└── evaluation.py

evals/
├── datasets/
│   ├── bengaluru.json
│   └── bengaluru_eval.json
└── results/
    └── generation_results_by_k.json

tests/
├── __init__.py
├── mock/
│   └── mock_generation_result.py
└── test_generation.py
```

## Setup

### Requirements

* Python 3.12+
* Ollama

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the project and development dependencies:

```bash
pip install -e ".[dev]"
```

Pull the required Ollama models:

```bash
ollama pull qwen3-embedding:0.6b
ollama pull qwen3.5:9b
```

Make sure Ollama is running before executing the RAG pipeline.

## Running Tests

```bash
pytest
```

## Status

The current milestone is a working, tested RAG MVP.

The project is being developed incrementally toward a production-oriented enterprise knowledge assistant.

## Roadmap

1. Working RAG MVP
2. Document ingestion
3. PostgreSQL + pgvector
4. Metadata filtering
5. Hybrid retrieval improvements
6. Evaluation expansion
7. FastAPI service
8. Observability
9. Security and document permissions
10. Docker and GCP deployment

