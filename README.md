# Autonomous Research & Report Agent

An autonomous AI agent that researches questions by combining its own private
knowledge base (RAG) with live web search and external APIs, then answers with
cited sources. Built on a stateful LangGraph agent loop behind a provider-agnostic
LLM layer — milestone by milestone, from scratch, to understand every component
rather than treat any of it as a black box.

## What it does today
- **Autonomous tool use**: given a question, a LangGraph agent decides which tools
  to call — its internal document store, live web search, or an external API — and
  combines the results into one cited answer. No hardcoded routing.
- **Plan-and-execute research**: for complex goals, a LangGraph pipeline drafts an
  ordered research plan, executes each step with the tool-using agent, critiques its
  own draft (reflection), does one targeted re-research pass for gaps, and synthesizes
  a cited report that flags its own limitations.
- **Three real tools**:
  - `search_knowledge_base` — the full hybrid RAG retriever over internal docs.
  - `web_search` — live web results via Tavily.
  - `get_exchange_rate` — an example external API call (currency conversion).
- **Talks to any LLM** through a provider-agnostic factory (`app/llm.py`) — swap
  models/providers by changing one config value, no code changes. (Proven in
  practice: migrated from a deprecated Groq model to `openai/gpt-oss-20b` with a
  one-line change.)
- **Structured output**: the model returns typed objects (via Pydantic).
- **Stateful agent (LangGraph)**: a StateGraph with a model node, a ToolNode, and a
  conditional edge that loops until the model stops calling tools — native tool
  calling, with a recursion limit as a safety cap. Built after a from-scratch
  version, so the framework is understood, not magic.
- **Persistent memory**: per-conversation memory via a LangGraph checkpointer; a
  SQLite run-history store (`app/history.py`) that logs every run.
- **RAG pipeline**: load → chunk → embed → store in Chroma → retrieve → generate a
  grounded, cited answer that says "I don't know" when the context lacks it.
- **Hybrid retrieval stack**: semantic (vector) + keyword (BM25) search, fused with
  hand-implemented Reciprocal Rank Fusion (RRF), cross-encoder reranking, and
  metadata filtering.

## Why hybrid retrieval
Keyword and semantic search fail in opposite ways: keyword nails exact terms but
misses synonyms; vector matches meaning but fuzzes rare codes. Fusing both hedges
against not knowing the query type in advance. A cross-encoder reranker then reads
the query and each candidate passage together for a sharper final ordering.

## Architecture at a glance
- **LangChain** provides the parts (model wrapper, tools, loaders, splitters);
  **LangGraph** provides the wiring (a stateful graph that loops, branches, persists).
- **Tools are how the agent acts**: each is a small, well-described function; the
  model reads the docstrings to choose which to call. Network tools use timeouts,
  return errors instead of raising, and restrict where they can go.
- **Two database families, on purpose**: a vector DB (Chroma) for semantic knowledge
  and a relational DB (SQLite) for run history — each chosen for the question it
  answers. Designed to graduate to Qdrant/pgvector and Postgres at scale.

## Stack
- **LLM**: Groq (`openai/gpt-oss-20b`) via a swappable factory
- **Agent framework**: LangGraph (StateGraph, ToolNode, checkpointer) + LangChain
- **Web search**: Tavily (agent-focused search API)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, free)
- **Vector DB**: Chroma (persisted locally)
- **Keyword / rerank**: `rank_bm25`, `sentence-transformers` cross-encoder
- **Persistence**: SQLite (run history + optional agent-state checkpoints)

## Setup

**Prerequisites:** Python 3.11+ and free API keys from [Groq](https://console.groq.com)
(the LLM) and [Tavily](https://tavily.com) (web search).

```bash
# 1. Clone and enter the project
git clone https://github.com/<your-username>/autonomous-research-agent.git
cd autonomous-research-agent

# 2. Create and activate a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
#    Copy .env.example to .env, then fill in GROQ_API_KEY and TAVILY_API_KEY
copy .env.example .env

# 5. Build the vector index from the documents in data/
python ingest.py
```

**Try it:**

```bash
python ask.py "How many vacation days do I get?"                 # RAG Q&A
python research_agent.py "What laptops can new hires choose?"    # tool-using agent
python research_pipeline.py "How does our NW-3000 warranty compare to typical routers?"  # research pipeline
```
## Status
Actively developed. Completed: environment + safe secret handling, first LLM call,
structured output, a hand-built agent loop, the RAG core, hybrid retrieval
(BM25 + vector + RRF + reranking + metadata filtering), databases + persistent
memory, a LangGraph rebuild with checkpointed memory, real tools with autonomous
tool choice (RAG, web search, external API), and a plan-and-execute research pipeline
with reflection and cited synthesis. Planned: multi-agent orchestration (a supervisor
coordinating specialist agents), a product UI, and a retrieval evaluation set to
measure improvements.