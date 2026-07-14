# Autonomous Research & Report Agent

An autonomous AI agent that answers questions from a private knowledge base using
Retrieval-Augmented Generation (RAG), with a hybrid retrieval stack and a
provider-agnostic LLM layer. Built milestone by milestone, from scratch, to
understand every component rather than treat any of it as a black box.

## What it does today
- **Talks to any LLM** through a provider-agnostic factory (`app/llm.py`) — swap
  Groq / Gemini / Ollama by changing one config value, no code changes.
- **Structured output**: the model returns typed objects (via Pydantic), not just
  prose — the foundation for agent decision-making.
- **Hand-built agent loop** (`agent_loop.py`): a from-scratch reason -> act ->
  observe loop with a tool and a step cap, implemented without a framework.
- **RAG pipeline**: load -> chunk -> embed -> store in Chroma -> retrieve ->
  generate a grounded, cited answer that says "I don''t know" when the context
  lacks the answer (anti-hallucination).
- **Hybrid retrieval stack**:
  - Semantic (vector) search over a local Chroma DB
  - Keyword search (BM25) for exact terms, codes, and IDs
  - Hybrid fusion via hand-implemented Reciprocal Rank Fusion (RRF)
  - Cross-encoder reranking for final precision
  - Metadata filtering to restrict search by source

## Why hybrid retrieval
Keyword and semantic search fail in opposite ways: keyword nails exact terms but
misses synonyms; vector matches meaning but fuzzes rare codes. Fusing both hedges
against not knowing the query type in advance. A cross-encoder reranker then reads
the query and each candidate passage together for a sharper final ordering — in
this project''s demo, reranking recovered a correct passage that naive fusion had
ranked last.

## Stack
- **LLM**: Groq (`llama-3.3-70b-versatile`) via a swappable factory
- **Framework**: LangChain (LangGraph planned)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, free)
- **Vector DB**: Chroma (persisted locally)
- **Keyword / rerank**: `rank_bm25`, `sentence-transformers` cross-encoder

## Setup
## Status
Actively developed. Completed: environment + safe secret handling, first LLM call,
structured output, hand-built agent loop, RAG core, and hybrid retrieval
(BM25 + vector + RRF + reranking + metadata filtering). Planned: RAG and web
search as agent tools, orchestration with LangGraph, and a retrieval evaluation
set to measure improvements.
