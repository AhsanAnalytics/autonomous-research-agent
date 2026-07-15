# Autonomous Research & Report Agent

An autonomous AI agent that researches questions by combining its own private
knowledge base (RAG) with live web search and external APIs, then answers with
cited sources — coordinated by a LangGraph agent (single or multi-agent), served
over a FastAPI backend, and usable from a browser UI. Built milestone by milestone,
from scratch, to understand every component rather than treat any of it as a black box.

![Demo: asking the agent a question and getting a cited answer in the browser](demo.png)

## What it does today
- **Clickable web app**: a Streamlit UI + FastAPI backend — type a question in the
  browser and get a cited report. The agent is served over HTTP (`POST /run`,
  `GET /runs`), so any front-end can consume it.
- **Autonomous tool use**: given a question, a LangGraph agent decides which tools to
  call — its internal document store, live web search, or an external API — and
  combines the results into one cited answer. No hardcoded routing.
- **Three real tools**:
  - `search_knowledge_base` — the full hybrid RAG retriever over internal docs.
  - `web_search` — live web results via Tavily.
  - `get_exchange_rate` — a live external API call (currency conversion), wired into
    the research agent with a timeout, host allow-list, and graceful error handling.
- **Plan-and-execute research**: for complex goals, a LangGraph pipeline drafts an
  ordered plan, executes each step with the tool-using agent, critiques its own draft
  (reflection), does one targeted re-research pass for gaps, and synthesizes a cited
  report that flags its own limitations.
- **Multi-agent supervisor**: a supervisor agent coordinates specialists — a Researcher
  (web + knowledge-base tools), an Analyst (reasoning), and a Writer (report) — routing
  between them via structured decisions until the report is done, with per-specialist
  error handling.
- **Provider-agnostic LLM**: swap models/providers by changing one config value, no code
  changes. (Proven in practice: migrated across Groq models — including a deprecation —
  with one-line changes.)
- **Persistent memory**: per-conversation memory via a LangGraph checkpointer; a SQLite
  run-history store (`app/history.py`) that logs every run and survives restarts.
- **Hybrid retrieval stack**: semantic (vector) + keyword (BM25) search, fused with a
  hand-implemented Reciprocal Rank Fusion (RRF), cross-encoder reranking, and metadata
  filtering. Grounded generation cites sources and says "I don't know" when the context
  lacks the answer.

## Why hybrid retrieval
Keyword and semantic search fail in opposite ways: keyword nails exact terms but misses
synonyms; vector matches meaning but fuzzes rare codes. Fusing both hedges against not
knowing the query type in advance. A cross-encoder reranker then reads the query and each
candidate passage together for a sharper final ordering.

## Architecture at a glance
- **UI → API → agent**: a thin Streamlit UI calls a FastAPI backend, which runs the
  agent. The UI holds no logic — the API is the stable contract, so the front-end could
  be swapped (React, Slack bot, etc.) without touching the agent.
- **LangChain** provides the parts (model wrapper, tools, loaders, splitters);
  **LangGraph** provides the wiring (a stateful graph that loops, branches, persists).
- **Tools are how the agent acts**: each is a small, well-described function; the model
  reads the docstrings to choose which to call. Network tools use timeouts, return
  errors instead of raising, and restrict where they can go.
- **Two database families, on purpose**: a vector DB (Chroma) for semantic knowledge and
  a relational DB (SQLite) for run history — each chosen for the question it answers.
  Designed to graduate to Qdrant/pgvector and Postgres at scale.

## Stack
- **LLM**: Groq (`openai/gpt-oss-120b`) via a swappable factory
- **Agent framework**: LangGraph (StateGraph, ToolNode, supervisor, checkpointer) + LangChain
- **Web search**: Tavily (agent-focused search API)
- **Embeddings**: `BAAI/bge-small-en-v1.5` (local, free)
- **Vector DB**: Chroma (persisted locally)
- **Keyword / rerank**: `rank_bm25`, `sentence-transformers` cross-encoder
- **Persistence**: SQLite (run history + optional agent-state checkpoints)
- **API / UI**: FastAPI + Streamlit

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

# 4. Add your API keys: copy .env.example to .env, then fill in the keys
copy .env.example .env

# 5. Build the vector index from the documents in data/
python ingest.py
```

**Run it — command line:**

```bash
python ask.py "How many days off do I get?"                 # RAG Q&A
python research_agent.py "What laptops can new hires choose?"    # tool-using agent
python research_pipeline.py "Compare our NW-3000 warranty to typical routers"  # research pipeline
python supervisor_agent.py "Compare our NW-3000 warranty to typical routers"   # multi-agent team
```

**Run it — web app (two terminals):**

```bash
# Terminal 1 — the API backend
python -m uvicorn app.server:app --port 8000

# Terminal 2 — the UI (opens in your browser at http://localhost:8501)
python -m streamlit run ui.py
```

## Status
Actively developed. Completed: environment + safe secret handling, first LLM call,
structured output, a hand-built agent loop, the RAG core, hybrid retrieval
(BM25 + vector + RRF + reranking + metadata filtering), databases + persistent memory,
a LangGraph rebuild with checkpointed memory, real tools with autonomous tool choice
(RAG, web search, external API), a plan-and-execute research pipeline with reflection,
a supervisor multi-agent system, and a FastAPI + Streamlit product wrapper. Planned:
an evaluation set to measure retrieval/agent quality, plus shipping polish (tests, CI,
Docker).