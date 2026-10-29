# Project status (handoff file)

Read this at the start of every session. Update it at the end of every step.
Spec: docs/architecture.md. Rules: CLAUDE.md.

## Done
- **Foundations:** config, secrets, logger, role-aware `get_llm`, usage tracker, models (commits `cab8302`, `5cd987b`). Code under `src/core`, `src/utils`.
- **Market layer (yfinance):** provider, TTL cache, retry/stale/sample fallback, `CachedProvider` in `src/data/market/`. Sample snapshot at `src/data/market_snapshot.json`.
- **RAG:** 60 articles in `src/data/knowledge_base` (counted with `find`); tuned settings 300/120, top_k 6, threshold 0.30 in `config.yaml`; held-out 10 questions with hit rate 1.00 per README table (10/10 is the README's claim; not re-run this session, not verified). Eval harness in `tests/rag_eval/`.
- **Finance Q&A agent:** `src/agents/finance_qa.py` (commit `8037c30`).
- **Portfolio agent:** `src/agents/portfolio.py`, metrics in `src/core/portfolio_metrics.py`, instrument_type/asset_class split, concurrent quotes, expense-ratio coverage (commits `3558497`, `1113286`, `99b0697`).
- **Market agent:** `src/agents/market.py`; metrics in `src/core/market_metrics.py`, request/fetch in `src/core/market_history.py`, models in `src/core/market_models.py`; role `market_summary` and block `market_analysis:` in `config.yaml`; `GraphState` now has typed `market_params` and `topic_filter`. `live_check_market_agent.py` is written but **not run** (not verified against the real provider and LLM). Review follow-ups done: short-history flag, `sma_windows` validation, payload `source`, concurrency test.
- **Goal Planning agent:** `src/agents/goal_planning.py`; math in `src/core/goal_metrics.py`, validation in `src/core/goal_inputs.py`, models in `src/core/goal_models.py`; role `goal_summary` and block `goal_planning:` in `config.yaml`; `GoalPayload` extended; config validation for the block in `src/utils/config.py`. `live_check_goal_planning.py` is written but **not run** (not verified against the real LLM). Spec: `docs/architecture.md` section 2.4.

## Next
- News, Tax agents.
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
- Market: a history shorter than the period (`market_analysis.short_history_tolerance_pct`, 15) is flagged per ticker (`period_is_short`, `span_days`), warned about, and the summary input carries `period_covered`. Metrics are still computed over the shorter span.
- Market: `MarketPayload.source` is the least live source among the tickers (order `live`, `cache`, `stale cache`, `sample data`; unknown counts as worst). The unused `quotes` field is removed.
- `sma_windows` is validated when the config loads (`src/utils/config.py`, `ConfigError`).
- Summary prompt (`market_analysis.summary_prompt`): one short paragraph per ticker, at most 3 numbers each, one closing sentence explaining moving average and drawdown. The number-guard rule is unchanged.
- yfinance `Pandas4Warning`: cause was `yfinance/__init__.py:45`, which adds a `default` DeprecationWarning filter for `^yfinance` when it is imported; `Pandas4Warning` is a DeprecationWarning and the provider imports yfinance lazily, so that filter landed in front of ours. Fix: `load_yfinance()` in `src/data/market/yfinance_provider.py` imports yfinance and then applies our filter; no yfinance upgrade needed (installed 1.7.0, pandas 3.1.0). Not verified against a live call (live checks not run).
- `CachedProvider`: no new lock. `TTLCache` already guards every read and write with a `threading.Lock`; a concurrency test at `CachedProvider` level was added. Two threads asking for the same uncached key can both call the live provider (a duplicate fetch, not a wrong result); left as is.
- Goal Planning: scenario returns (low 4/6/8, medium 5/9/12, high 3/11/15 percent) and inflation (5 percent) are illustrative figures chosen by the project author, labelled with `as_of` 2026-10-29 and `source` in config; not forecasts. Source string deliberately names no external body.
- Goal Planning: annual effective returns converted to monthly `(1+r)^(1/12)-1`; deposits at month end; step-up applied every 12 months (from month 13). These appear in the payload's `assumptions` block and in architecture 2.4.
- Goal Planning: required monthly contribution uses the linear closed form `(target - FV(0)) / FV_unit`, exact with or without step-up, instead of a search (round-trip tested in `tests/test_goal_metrics.py`).
- Goal Planning: `inflation_adjust` reads `goal_amount` as today's money; the target is the goal in future money, and shortfall, required contribution and months use that target. Default from `goal_planning.inflation_adjust_default` (true); `goal_params["inflation_adjust"]` overrides.
- Goal Planning: LLM role is `goal_summary` (the doc said `default`; doc changed). The number guard is still imported from `portfolio_metrics`.
- Goal Planning: commentary input uses western digit grouping (`1,234,567.00`).

## Open to-dos
- UI formats amounts in Indian lakh/crore grouping while the LLM input stays western grouping; the number guard (`numbers_in`) does not parse lakh grouping.
- Run `live_check_goal_planning.py` once (spends API money; only when asked).
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
- At the workflow step, run the number check on text reworded by the output guard (`guard_answer` rewording is not checked today).
- Once the synthesizer is wired, verify the disclaimer end to end (agents return answers without it; not verified for any agent today).
- README known limitation to add: the number guard accepts any number present anywhere in the input, so a coincidental match is not detected.

## Last updated
2026-10-29, Goal Planning agent: pure projection math, guarded commentary, assumptions block (see `git log -1` for the commit).
