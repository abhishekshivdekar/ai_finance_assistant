# Project status (handoff file)

Read this at the start of every session. Update it at the end of every step.
Spec: docs/architecture.md. Rules: CLAUDE.md.

## Done
- **Foundations:** config, secrets, logger, role-aware `get_llm`, usage tracker, models (commits `cab8302`, `5cd987b`). Code under `src/core`, `src/utils`.
- **Market layer (yfinance):** provider, TTL cache, retry/stale/sample fallback, `CachedProvider` in `src/data/market/`. Sample snapshot at `src/data/market_snapshot.json`.
- **RAG:** 60 articles in `src/data/knowledge_base` (counted with `find`); tuned settings 300/120, top_k 6, threshold 0.30 in `config.yaml`; held-out 10 questions with hit rate 1.00 per README table (10/10 is the README's claim; not re-run this session, not verified). Eval harness in `tests/rag_eval/`.
- **Finance Q&A agent:** `src/agents/finance_qa.py` (commit `8037c30`).
- **Portfolio agent:** `src/agents/portfolio.py`, metrics in `src/core/portfolio_metrics.py`, instrument_type/asset_class split, concurrent quotes, expense-ratio coverage (commits `3558497`, `1113286`, `99b0697`).

## Next
- Market agent (trend analysis), Goal Planning, News, Tax agents.
- Router, LangGraph workflow, clarify node, facade.
- Persistence.
- Streamlit UI (`src/web_app/` only has `__init__.py` today).
- README completion, benchmarks, demo video.
- Optional: Docker / MCP.

## Decisions
- Portfolio: `instrument_type` (what it is) is separate from `asset_class` (what it behaves like); risk, allocation and the single-class penalty use `asset_class` only (`docs/architecture.md` section 2.2).
- RAG settings chosen by the eval harness (README "RAG tuning results").
- Google News RSS: **not in `src/`**. It appears only in the root `debug_news.py` (a scratch script). The "market layer includes Google News RSS" claim is not verified; the News agent will need it built.

## Open to-dos
- Resolve the `<!-- VERIFY -->` markers in the knowledge base against official sites: **21 markers in 15 files** (counted with grep).
- Add `topic_filter` to `GraphState` (`src/workflow/state.py` has no such key; `src/agents/finance_qa.py` already reads `state.get("topic_filter")`).
- Portfolio tab needs a per-holding asset-class dropdown (UI not built yet).
- UI shows sources per chunk with article title and snippet.
- Delete `debug_news.py` (exists at the repo root).
- Review `smoke_test.py` (exists at the repo root).
- Document tech choices in the README: FAISS vs Pinecone, yfinance vs Alpha Vantage, OpenAI vs Gemini, Google News RSS.
- Rotate the OpenAI key if not done (not verified; cannot be checked from the repo).
- Pin `requirements.txt` at the end (all entries are unpinned today).

## Last updated
2026-10-29, after commit `99b0697` (this file is added in the next commit; see `git log`).
