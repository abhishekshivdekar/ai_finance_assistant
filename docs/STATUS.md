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
- **News agent:** `src/agents/news.py`, Google News RSS source in `src/data/news/`; see Decisions.
- **Goal Planning agent:** `src/agents/goal_planning.py`; math in `src/core/goal_metrics.py`, validation in `src/core/goal_inputs.py`, models in `src/core/goal_models.py`; role `goal_summary` and block `goal_planning:` in `config.yaml`; `GoalPayload` extended; config validation for the block in `src/utils/config.py`. `live_check_goal_planning.py` is written but **not run** (not verified against the real LLM). Spec: `docs/architecture.md` section 2.4. Review follow-ups done (huge-number validation, number-guard whitelist, strings moved to config, negative-rate config check).

## Next
- Tax agent.
- Router, LangGraph workflow, clarify node, facade.
- Persistence.
- Streamlit UI (`src/web_app/` only has `__init__.py` today).
- README completion, benchmarks, demo video.
- Optional: Docker / MCP.

## Decisions
- News agent: `src/agents/news.py` (`NewsAgent`, role `news_summary`, block `news:` in `config.yaml`); pure parsing in `src/core/news_parse.py`, query resolution in `src/core/news_query.py`, shared citation helpers in `src/core/citations.py` (`apply_citations` moved out of `finance_qa.py`, behaviour unchanged), `NewsPayload` in `src/core/models.py`, `news_params` typed key in `GraphState`, `_validate_news` in `src/utils/config.py`. Spec: architecture 2.5 and section 6. `live_check_news.py` is written but **not run** (not verified against the real feed and LLM).
- News: `get_news` now takes a search phrase (`validate_news_query`), not a ticker; the yfinance news path (`_parse_news`, the `news` branch of `_yf_call`) is removed with its tests. `YFinanceProvider(news_source=...)` delegates to `GoogleNewsRssSource`, so `CachedProvider` gives news cache, retry, stale cache and errors with no change to `fallback.py`. Provider error classes moved to `src/data/errors.py` (still importable from `provider.py`) to avoid a circular import.
- News: invalid `[n]` citations are dropped, not renumbered (the payload always lists every headline with its number); a summary with no valid citation falls back; the number guard removes `[n]` markers first and allows numbers from headline titles and publishers only; the query wins over the ticker; provider failure is `no_answer`; only http(s) links; DTD/entity feeds rejected; news TTL reuses `market_data.cache_ttl_minutes`; ticker phrases come from `news.query_overrides` then `news.query_templates` (the market provider offers no cheap company name).
- Goal Planning: a step-up above `max_step_up_pct` is rejected with `needs_input` (message `step_up_range`), not capped; `docs/architecture.md` 2.4 now says so.
- Goal Planning: numbers are validated without `float()`/`math.isfinite()` on huge ints (`_is_number` in `src/core/goal_inputs.py`): the limit is compared as an int first, so `10**400` gives `too_large` / horizon-range / step-up-range messages; `inf`/`nan` give "must be a number"; `-10**400` is clamped to `-limit` so the negative-value message shows.
- Goal Planning: the number guard's whitelist leaves out the `assumptions` entry (as_of, timing notes, source), so an invented "10 years" or "13 months" is rejected; computed values and the user's inputs still pass. The prompt asks the LLM not to repeat numbers from the notes.
- Goal Planning: the three commentary warnings and the "not applied" text moved to `goal_planning.texts`; the summary horizon now uses `years_months_text` (so 1 year reads "1 year").
- Goal Planning bug fix: with inflation on, `months_to_goal` compares month k with `goal_today x (1+i)^(k/12)` (`months_to_goal(..., inflation_pct)` in `src/core/goal_metrics.py`), not with the target fixed at the horizon. Shortfall, `goal_met` and required contribution are unchanged (measured at the horizon). Live-check goal 1 gives 488 / 283 / 223 months (conservative / expected / optimistic), tested. `time_compared_with_horizon` wording left as is. Documented in architecture 2.4.
- Goal Planning below-inflation fix: `months_to_goal` is the first crossing only. The rising goal uses `(1+i)^(k/12)`, equal to the horizon target at month `horizon x 12`; with a return below inflation a balance can cross and later fall back below (also: savings above today's goal but below the inflated target gave "already met" next to a shortfall). Such scenarios get `below_inflation` (strictly below; `ScenarioResult` in `src/core/goal_models.py`, set in `_scenario` in `src/core/goal_metrics.py`), a `below_inflation_months_note`, a `below_inflation_text` in the commentary input and a payload warning (`src/agents/goal_planning.py`). Every scenario gets a precomputed `status_text` (`status_text()` in `goal_metrics.py`, six templates in `goal_planning.texts.status_*`, no "reached" wording) stating both facts. With inflation on, `summary_input` carries two more labelled sentences (goal price at the end of the horizon; goal price keeps rising). Maths unchanged. Architecture 2.4 updated. The 488 / 283 / 223 test now passes returns and inflation explicitly (`explicit_cfg` in `tests/test_goal_metrics.py`) and is renamed.
- Config load rejects negative scenario returns and negative inflation (`src/utils/config.py`).
- Portfolio: `Holding` now rejects `inf` for `quantity`, `avg_cost`, `expense_ratio_pct` (`allow_inf_nan=False`); before, `inf` passed the `gt=0` / `ge=0` checks. Huge ints and `nan` were already rejected by pydantic. Market `parse_request` cannot raise on huge/inf/nan values (regression test added); it only echoes odd values in warnings.
- Portfolio: `instrument_type` (what it is) is separate from `asset_class` (what it behaves like); risk, allocation and the single-class penalty use `asset_class` only (`docs/architecture.md` section 2.2).
- RAG settings chosen by the eval harness (README "RAG tuning results").
- News: Google News RSS now lives in `src/data/news/google_rss.py` (`GoogleNewsRssSource`, a `NewsSource` Strategy injected into `YFinanceProvider`); `debug_news.py` is deleted. The earlier claim "market layer includes Google News RSS" was wrong until this step.
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
- Goal Planning: the required monthly amount is labelled in the payload (`required_labels`, per-scenario `required_label`): plain, "starting monthly amount (rises X% a year)" with a step-up, "minimum monthly amount to still reach the goal" when the scenario already reaches the goal (here "reaches" means final value >= target at the current contribution).
- Goal Planning: commentary input has precomputed text (`years_months_to_goal`, `surplus_or_shortfall_text`, `time_compared_with_horizon`, two plain inflation sentences); the old `labels` object is no longer sent to the LLM. Maths unchanged.

## Open to-dos
- UI formats amounts in Indian lakh/crore grouping while the LLM input stays western grouping; the number guard (`numbers_in`) does not parse lakh grouping.
- Run `live_check_goal_planning.py` once (spends API money; only when asked).
- Resolve the `<!-- VERIFY -->` markers in the knowledge base against official sites: **21 markers in 15 files** (counted with grep).
- Portfolio tab needs a per-holding asset-class dropdown (UI not built yet).
- UI shows sources per chunk with article title and snippet.
- Move the number guard (`numbers_in` / `collect_numbers`) out of `portfolio_metrics.py` into a shared module used by Portfolio and Market.
- Run `live_check_news.py` once (spends API money and uses the network; only when asked).
- Run `live_check_market_agent.py` once (spends API money; only when asked).
- Review `smoke_test.py` (exists at the repo root).
- Document tech choices in the README: FAISS vs Pinecone, yfinance vs Alpha Vantage, OpenAI vs Gemini, Google News RSS (chosen because the yfinance news path returned nothing usable per the project author, and the RSS feed supports free search phrases like "Nifty 50"; stdlib only, no new dependency; see architecture section 6).
- Rotate the OpenAI key if not done (not verified; cannot be checked from the repo).
- Pin `requirements.txt` at the end (all entries are unpinned today).
- At the workflow step, run the number check on text reworded by the output guard (`guard_answer` rewording is not checked today).
- Once the synthesizer is wired, verify the disclaimer end to end (agents return answers without it; not verified for any agent today).
- README known limitation to add: the number guard accepts any number present anywhere in the input, so a coincidental match is not detected.

## Last updated
2026-10-29, News agent (Google News RSS source, guarded summary, tests; see `git log -1` for the commit).
