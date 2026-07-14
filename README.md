# Autonomous Research & Report Agent

An autonomous AI agent that answers questions from a private knowledge base using
Retrieval-Augmented Generation (RAG), a hybrid retrieval stack, and a stateful
LangGraph agent loop — all behind a provider-agnostic LLM layer. Built milestone by
milestone, from scratch, to understand every component rather than treat any of it
as a black box.

## What it does today
- **Talks to any LLM** through a provider-agnostic factory (`app/llm.py`) — swap
  Groq / Gemini / Ollama by changing one config value, no code changes.
- **Structured output**: the model returns typed objects (via Pydantic), not just
  prose — the foundation for agent decision-making.
- **Stateful agent (LangGraph)**: a `StateGraph` with a model node, a `ToolNode`, and
  a conditional edge that loops until the model stops calling tools — using native
  tool calling and a recursion limit. Built *after* a from-scratch version
  (`agent_loop.py`), so the framework is understood, not magic.
- **Persistent memory**:
  - Per-conversation memory via a LangGraph checkpointer keyed by `thread_id`.
  - A SQLite run-history store (`app/history.py`) that logs every run and survives
    across restarts.
- **RAG pipeline**: load -> chunk -> embed -> store in Chroma -> retrieve -> generate
  a grounded, cited answer that says "I don''t know" when the context lacks the answer
  (anti-hallucination).
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
the query and each candidate passage together for a sharper final ordering — in this
project''s demo, reranking recovered a correct passage that naive fusion had ranked
last.

## Architecture at a glance
- **LangChain** provides the parts (model wrapper, tools, loaders, splitters);
  **LangGraph** provides the wiring (a stateful graph that loops, branches, persists).
- **Two database families, on purpose**: a vector DB (Chroma) for semantic knowledge
  and a relational DB (SQLite) for structured run history — each chosen for the
  question it answers. Designed to graduate to Qdrant/pgvector (vectors) and
  Postgres (relational) at scale without rewrites.

## Stack
- **LLM**: Groq (`llama-3.3-70b-versatile`) via a swappable factory
- **Agent framework**: LangGraph (StateGraph, ToolNode, checkpointer) + LangChain
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, free)
- **Vector DB**: Chroma (persisted locally)
- **Keyword / rerank**: `rank_bm25`, `sentence-transformers` cross-encoder
- **Persistence**: SQLite (run history + optional agent-state checkpoints)

## Setup
## Status
Actively developed. Completed: environment + safe secret handling, first LLM call,
structured output, a hand-built agent loop, the RAG core, hybrid retrieval
(BM25 + vector + RRF + reranking + metadata filtering), databases + persistent
memory, and a LangGraph rebuild with checkpointed memory. Planned: RAG and web
search as agent tools, plan-and-execute research, a multi-agent supervisor, a
product UI, and a retrieval evaluation set to measure improvements.
