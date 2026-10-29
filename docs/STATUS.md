# Project status (handoff file)

Read this at the start of every session. Update it at the end of every step.
Spec: docs/architecture.md. Rules: CLAUDE.md.

## Done
- **Foundations:** config, secrets, logger, role-aware `get_llm`, usage tracker, models (commits `cab8302`, `5cd987b`). Code under `src/core`, `src/utils`.
- **Market layer (yfinance):** provider, TTL cache, retry/stale/sample fallback, `CachedProvider` in `src/data/market/`. Sample snapshot at `src/data/market_snapshot.json`.
- **RAG:** 60 articles in `src/data/knowledge_base` (counted with `find`); tuned settings 300/120, top_k 6, threshold 0.30 in `config.yaml`; held-out 10 questions with hit rate 1.00 per README table (10/10 is the README's claim; not re-run this session, not verified). Eval harness in `tests/rag_eval/`.
- **Finance Q&A agent:** `src/agents/finance_qa.py` (commit `8037c30`).
- **Portfolio agent:** `src/agents/portfolio.py`, metrics in `src/core/portfolio_metrics.py`, instrument_type/asset_class split, concurrent quotes, expense-ratio coverage (commits `3558497`, `1113286`, `99b0697`).
- **Market agent:** `src/agents/market.py`; metrics in `src/core/market_metrics.py`, request/fetch in `src/core/market_history.py`, models in `src/core/market_models.py`; role `market_summary` and block `market_analysis:` in `config.yaml`; `GraphState` now has typed `market_params` and `topic_filter`. `live_check_market_agent.py` is written but **not run** (not verified against the real provider and LLM).

## Next
- Goal Planning, News, Tax agents.
- Router, LangGraph workflow, clarify node, facade.
- Persistence.
- Streamlit UI (`src/web_app/` only has `__init__.py` today).
- README completion, benchmarks, demo video.
- Optional: Docker / MCP.

## Decisions
- Portfolio: `instrument_type` (what it is) is separate from `asset_class` (what it behaves like); risk, allocation and the single-class penalty use `asset_class` only (`docs/architecture.md` section 2.2).
- RAG settings chosen by the eval harness (README "RAG tuning results").
- Google News RSS: **not in `src/`**. It appears only in the root `debug_news.py` (a scratch script). The "market layer includes Google News RSS" claim is not verified; the News agent will need it built.
- Market: one 1y history call per ticker is sliced in code; period return, volatility and drawdown use the requested period, SMAs and the 52-week range use the full year, so the trend label reflects the latest averages. Day change comes from the last two closes (no `get_quote`). Documented in `docs/architecture.md` section 2.3.
- Market: the number guard is imported from `portfolio_metrics` for now.

## Open to-dos
- Resolve the `<!-- VERIFY -->` markers in the knowledge base against official sites: **21 markers in 15 files** (counted with grep).
- Portfolio tab needs a per-holding asset-class dropdown (UI not built yet).
- UI shows sources per chunk with article title and snippet.
- Move the number guard (`numbers_in` / `collect_numbers`) out of `portfolio_metrics.py` into a shared module used by Portfolio and Market.
- Run `live_check_market_agent.py` once (spends API money; only when asked).
- Delete `debug_news.py` (exists at the repo root).
- Review `smoke_test.py` (exists at the repo root).
- Document tech choices in the README: FAISS vs Pinecone, yfinance vs Alpha Vantage, OpenAI vs Gemini, Google News RSS.
- Rotate the OpenAI key if not done (not verified; cannot be checked from the repo).
- Pin `requirements.txt` at the end (all entries are unpinned today).

## Last updated
2026-10-29, Market agent step (see `git log -1` for the commit).
