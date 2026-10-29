# AI Finance Assistant: Architecture

Scope tags: **[MVP]** = built in the 2-week scope. **[Later]** = designed for, not built now.
Agents use a separate **build status** (build first / build after); all six are designed into the graph now.
All settings live in `config.yaml` and are read via `get_config()`; secrets via `get_secret()`.
Model IDs live in `config.yaml` and are verified by `python smoke_test.py`.

---

## 1. Component diagram

```mermaid
flowchart TD
    subgraph UI["Streamlit UI (src/web_app)"]
        T1[Chat tab]
        T2[Portfolio tab]
        T3[Market tab]
        T4[Goals tab]
        SB[Sidebar: profile name, tokens, LangSmith status]
    end

    T1 --> F[Workflow facade run_query]
    T2 & T3 & T4 -->|intent_hint| F
    F --> G

    subgraph G["LangGraph StateGraph (src/workflow)"]
        S0([START]) -->|conditional: intent_hint?| R
        S0 -->|hint present| QA & PA & MA & GP & NW & TX
        R[router node + confidence]
        QA[finance_qa]
        PA[portfolio]
        MA[market]
        GP[goal_planning]
        NW[news]
        TX[tax_education]
        CL[clarify node]
        SR[safe_response]
        SY[synthesizer]
        FB[fallback]
        R -->|conditional| QA & PA & MA & GP & NW & TX & SR & CL
        QA & PA & MA & GP & NW & TX -->|conditional on status| SY
        QA & PA & MA & GP & NW & TX -.re-route max 2 hops.-> QA & MA
        QA & PA & MA & GP & NW & TX -.needs_input.-> CL
        QA & PA & MA & GP & NW & TX -.all failed.-> FB
        R -.error.-> FB
        CL --> E([END])
        SR --> E
        SY --> E
        FB --> E
    end

    G <--> CP[(Checkpointer: SQLite)]
    PA & GP --> PR[ProfileRepository]
    PR --> DB[(SQLite profile store)]
    CP --- DB

    QA --> RAG[(RAG: FAISS + MiniLM)]
    TX --> RAG
    TX --> TC[Tax calculators, rates from config.yaml]
    MA --> MD[Market data layer: CachedProvider, retry, fallback]
    PA --> MD
    NW --> MD
    GP --> CALC[Projection calculators]
    MD --> YF[(yfinance)]
    MD --> SNAP[(Bundled sample snapshot)]

    G --> LF[get_llm role=...]
    LF --> LLM[(LLM provider)]
    G -.optional traces.-> LS[(LangSmith tracer)]
    G --> CFG[get_config / get_secret / get_logger]
```

**Why:** the UI only talks to one facade, the graph only talks to agents, and agents only reach the outside world through tools, the repository and `get_llm()`. Each layer can be mocked in tests.

---

## 2. Agents

All agents extend `BaseAgent` (section 3). Every agent provides information and education. None gives a personalised buy/sell/hold recommendation.

| # | Agent | Build status | Deterministic tool |
|---|---|---|---|
| 2.1 | Finance Q&A | build first | none (RAG retriever) |
| 2.2 | Portfolio Analysis | build first | metric functions |
| 2.3 | Market Analysis | build first | market data layer |
| 2.4 | Goal Planning | build first | compound-growth calculator |
| 2.5 | News Synthesizer | built | Google News RSS headlines (stdlib) |
| 2.6 | Tax Education (India) | build after | illustrative tax calculators |

**Market scope:** US tickers (`AAPL`) and NSE tickers with the `.NS` suffix (`RELIANCE.NS`). Every price and portfolio value is shown with its currency (USD or INR); values in different currencies are never summed without conversion or a clear per-currency split. The bundled sample snapshot and the sample portfolios include both US and Indian examples.

### 2.1 Finance Q&A
- **Purpose:** answer general finance-education questions with cited sources.
- **Inputs:** user query, recent `messages`.
- **Outputs:** answer text, list of `Source(title, url, chunk_id, score)`.
- **Uses:** RAG retriever (section 5), two LLM calls per question. Retrieval covers all topics unless the Chat tab's `topic_filter` is set.
- **Pipeline (`src/agents/finance_qa.py`, prompts and messages in `config.yaml` under `finance_qa:`):**
  1. **Rewrite:** one cheap call turns a colloquial question into a short finance-term query. On any failure, an empty reply or a malformed one (no letters, far over `rewrite_max_chars`), the original question is used; rewriting never fails the request.
  2. **Retrieve** with the original and the rewritten query (`retrieve()`), each also searched with its acronym-expanded form when it holds one (`expand_query()`, table `rag.query_expansions`: whole words, any case, no LLM call; e.g. `LTCG` -> "long-term capital gains tax", because the articles use the long form), merge by chunk keeping the best score, then re-apply the shared `select_chunks()` rules and `rag.top_k` to the merged list.
  3. **Nothing retrieved:** `status="no_answer"` with a fixed message, and the answer LLM is not called.
  4. **Grounded answer:** one call whose system prompt says to answer only from the numbered context, cite `[1]`, `[2]`, reply exactly `NO_ANSWER` when the context does not actually answer (even for finance topics the knowledge base lacks, such as loans or crypto), stay educational, give no tax figures from memory, and use short beginner language.
  5. **Parse:** `NO_ANSWER` gives `no_answer` with the same fixed message. Otherwise `status="ok"`; citations that do not exist are removed, the rest are renumbered `1..m`, and `sources` holds only the chunks actually cited, in that order.
- **LLM roles:** `finance_qa_rewrite` and `finance_qa` (both merged over `llm.*`), so usage is recorded per agent and per call type.
- **Payload:** `QAPayload` with `rewritten_query` and `retrieved` (count). An empty query returns `needs_input`.

### 2.2 Portfolio Analysis
- **Purpose:** explain a user's holdings: allocation, concentration, diversification and a simple risk label. Educational only.
- **Principle:** every number is computed in deterministic Python (`src/core/portfolio_metrics.py`). The agent makes one LLM call, to write a short plain-language summary from the already-computed metrics, and the LLM never calculates or invents a number. The output guard in `BaseAgent.run()` (section 3.3) makes one more call to reword the answer if a guard pattern matches, so at most two calls happen per request.
- **Inputs:** `state["portfolio"]`, a list of holdings: `ticker`, `quantity` (> 0), optional `avg_cost` (per unit, assumed to be in the quote's currency), optional `instrument_type` (`stock`, `etf`, `fund`, `bond`, `cash`, `other`; what the holding is; default `stock`), optional `asset_class` (`equity`, `debt`, `gold`, `cash`, `other`; what it behaves like; default `equity`) and optional `expense_ratio_pct` (annual fund cost in percent, e.g. `0.20`; never guessed when missing). The currency is not an input: it comes from the quote. Models are immutable pydantic classes in `src/core/portfolio_models.py` (`Holding`, `Position`, `CurrencyMetrics`, `PortfolioMetrics`).
- **Valuation (`src/core/portfolio_valuation.py`):** each holding is priced through the injected market provider (the shared `CachedProvider`, so cache, retry, stale-cache and sample-data fallbacks apply). Currencies are **never added or converted** and no FX rate is invented: totals, weights and every metric below are per currency. Quotes are fetched concurrently in a `ThreadPoolExecutor` (at most `portfolio.max_workers` threads, results kept in input order, one failing ticker never affects the others). A holding whose price is unavailable is left out of the totals and named in the warnings, and the rest is still analysed. Prices from stale cache or sample data are flagged on the position and in the warnings. Invalid rows, duplicate tickers and holdings beyond `portfolio.max_holdings` are skipped with a warning.
- **Metrics (pure functions; thresholds in `config.yaml` under `portfolio:`, labelled "illustrative"):**
  - **Weights:** position value divided by its currency's total (sums to 1 per currency).
  - **Unrealised P&L and %:** only where `avg_cost` was given: `(price - avg_cost) x quantity` and `(price / avg_cost - 1) x 100`.
  - **Allocation by asset class:** share of value per `asset_class`. `instrument_type` is never used for allocation, equity share, risk or the diversification penalty: NIFTYBEES.NS is `etf` by type but `equity` by class.
  - **Concentration:** largest position weight and top-3 weight. A warning is added when the largest weight reaches `concentration_warning`, and a single-holding portfolio gets its own warning.
  - **Diversification score (0 to 100):** `HHI = sum(weight^2)`; `score = clamp(100 x (1 - HHI) / (1 - 1/N_full), 0, 100)`, minus `single_class_penalty` points when every position has the same `asset_class` (instrument type is ignored), floored at 0. `N_full = full_diversification_holdings` is the number of equal positions that counts as fully spread. Examples with `N_full = 20` and penalty 15: one holding scores 0; five equal equities score round(100 x 0.8 / 0.95) = 84, minus 15 = 69; four equal equity positions, one of them an ETF, score round(100 x 0.75 / 0.95) = 79, minus 15 = 64 (`Moderate`, not `Well diversified`). Label from `diversification_bands`: `Concentrated` (below `concentrated_below`), `Well diversified` (from `diversified_from`), otherwise `Moderate`. It is an educational indicator, not a rating.
  - **Risk label (`Low` / `Moderate` / `High`):** the higher of two levels. The equity level uses the value share whose `asset_class` is in `equity_like_classes` (`equity`, `other`; an unknown class counts as risky, so the label errs cautious) against `equity_moderate_from` and `equity_high_from`. The concentration level uses the largest weight against `concentration_moderate_from` and `concentration_high_from`. It describes the mix only, not a personal risk assessment.
  - **Weighted-average expense ratio:** over the holdings that provide one, with weights re-normalised to those holdings; coverage is reported as share of portfolio value and holding count ("covers 47% of value, 1 of 4 holdings"); "not provided", with no coverage fraction, when none do.
- **Summary step:** one call with the `portfolio_summary` role. The prompt (`portfolio.summary_prompt`) receives only the computed metrics as JSON with every number pre-formatted as text (`summary_input`), and asks for beginner language, observation wording ("the largest holding is 42.0% of the INR portfolio"), no buy/sell/hold advice, and mentions of concentration, diversification, the risk label and expense-ratio coverage. A **number guard** rejects a summary containing any number that is not in that JSON. If the call fails, the reply is empty or the guard rejects it, the fixed `portfolio.summary_fallback` text is used, `status` stays `ok` and a warning is added: the summary can never fail the request.
- **Outputs:** `status="ok"` with `answer` = the summary and `data` = `PortfolioPayload` (`metrics`: the `PortfolioMetrics` as a dict with `positions`, `currencies`, `warnings`, `as_of`, `uses_stale_or_sample`; and `warnings`). An empty or fully invalid portfolio gives `needs_input`; if no holding can be priced, `no_answer` (the LLM is not called).
- **Uses:** market data layer for prices, pure-Python metric functions, LLM role `portfolio_summary` (own role so usage is counted per agent).

### 2.3 Market Analysis
- **Purpose:** explain recent price behaviour (trend, volatility, drawdown, position against moving averages) of one or more tickers or indices. Educational only: observations, never predictions or buy/sell/hold wording.
- **Principle:** every number is computed in deterministic Python (`src/core/market_metrics.py`). The agent makes one LLM call, to write a short plain-language summary from the already-computed metrics, and the LLM never calculates or invents a number. The output guard in `BaseAgent.run()` (section 3.3) makes one more call to reword the answer if a guard pattern matches, so at most two calls happen per request.
- **Inputs:** `state["market_params"] = {"tickers": [...], "period": "6mo"}` (set by the Market tab or the router). Tickers are validated with `validate_symbol`, de-duplicated and capped at `market_analysis.max_tickers`; the period is one of `market_analysis.periods` (`1mo`, `3mo`, `6mo`, `1y`) and falls back to `default_period` with a warning. Invalid tickers are skipped with a warning; no valid ticker gives `needs_input`.
- **Data (`src/core/market_history.py`):** one `get_history(ticker, fetch_period)` call per ticker (`fetch_period: 1y`) through the injected provider (the shared `CachedProvider`, so cache, retry, stale-cache and sample-data fallbacks apply), fetched concurrently (`ThreadPoolExecutor`, at most `market_analysis.max_workers` threads, input order kept, one failing ticker never affects the others). The same year of daily closes is **sliced in code** to the requested period (`period_days`: the rows within that many calendar days of the last row). Each ticker is reported in its own currency and nothing is added across tickers or currencies.
- **Metrics (pure functions; thresholds in `config.yaml` under `market_analysis:`, labelled "illustrative"):**
  - **Latest price and day change** (full year): last close, and `(last - previous close)` with its percent of the previous close. There is no separate `get_quote` call.
  - **Period return** (requested period slice): `(last close / first close - 1) x 100`.
  - **Simple moving averages** (full year, `sma_windows`, default 20 and 50 trading days): mean of the last `w` closes, and `price vs SMA = (price / SMA - 1) x 100`. A window longer than the history is omitted with a warning. `sma_windows` must be exactly two distinct positive integers, short first (checked when the config is loaded; `ConfigError` otherwise).
  - **Annualised volatility** (period slice): sample standard deviation of the daily simple returns x `sqrt(trading_days)` x 100 (`trading_days: 252`); needs at least three closes.
  - **Maximum drawdown** (period slice): the largest fall from a running peak, `min(close / running peak - 1) x 100` (0 or negative).
  - **52-week range** (full year, closes): high, low, `% below high = (1 - price / high) x 100` and `% above low = (price / low - 1) x 100`. Omitted with a warning only when the fetched history spans fewer than `week52_min_days` calendar days (for example a recent listing), whatever period was requested.
  - **Trend label:** `Uptrend` if the short average is above the long one by more than `trend.sma_gap_pct` percent and the period return is at least `+trend.return_threshold_pct`; `Downtrend` for the mirror image; otherwise `Sideways`. The averages describe the latest data, so the label reflects the latest averages and the selected period's return, not only the selected window. Omitted with a warning if an average is unavailable.
  - **Short history:** the period metrics (return, volatility, drawdown) use the slice of closes within `period_days` of the last close. If that slice spans fewer calendar days than `period_days x (1 - short_history_tolerance_pct / 100)`, the ticker is flagged (`period_is_short`, `span_days` in `TickerMetrics`), a warning reads "only N days of data for the requested 6mo; return, volatility and drawdown cover that shorter span", and the summary input carries `period_covered` so the model does not call the figure a return over the whole period (for example the 5-close sample snapshot with a 6-month request).
  - A metric that cannot be computed (too few closes) is `None` ("not available") with a warning; it never fails the ticker. An unavailable ticker is a warning and the others are still processed; if none can be analysed the status is `no_answer` and the LLM is not called. Stale cache and sample prices are flagged on the ticker and in the warnings.
  - **Comparison table** (more than one ticker): return, volatility, drawdown and trend per ticker in input order, with no ranking.
- **Summary step:** one call with the `market_summary` role. The prompt (`market_analysis.summary_prompt`) receives only the computed metrics as JSON with every number pre-formatted as text (`summary_input`; label numbers such as "20-day average" and "52-week high" are inside the values so the model may repeat them), and asks for one short paragraph per ticker with at most 3 numbers each (period return, position against the 20/50-day averages with one percentage, distance from the 52-week high), plain beginner language, no repeated metrics or semicolon lists, observation wording ("the price is below its short-term average"), no predictions and no buy/sell/hold wording, and one closing sentence that explains a moving average and a drawdown once. The **number guard** (`numbers_in` / `collect_numbers`, shared with the Portfolio agent) rejects a summary containing any number that is not in that JSON. If the call fails, the reply is empty or the number guard rejects it, the fixed `market_analysis.summary_fallback` text is used, `status` stays `ok` and a warning is added. Only this step falls back; any other exception is turned into an `error` result by `BaseAgent.run()`. Text reworded later by the output guard is not number-checked yet (open to-do in `docs/STATUS.md`).
- **Outputs:** `status="ok"` with `answer` = the summary and `data` = `MarketPayload` (`metrics`: `MarketMetrics` as a dict with `period`, `tickers`, `warnings`, `as_of`, `uses_stale_or_sample`; `comparison`; `history`: flat `{ticker, date, close}` rows of the selected period for charts; `as_of`; `source`: the least live source among the analysed tickers in the order `live`, `cache`, `stale cache`, `sample data` (an unknown label counts as the worst); `is_stale`; `warnings`). No valid ticker gives `needs_input`; nothing analysable gives `no_answer`.
- **Uses:** market data layer for history, pure-Python metric functions, LLM role `market_summary` (own role so usage is counted per agent).

### 2.4 Goal Planning
- **Purpose:** risk-aware, illustrative projections toward a goal (retirement, house, etc.). Never a prediction or a promise of returns.
- **Principle:** every number is computed in deterministic Python (`src/core/goal_metrics.py`). The agent makes one LLM call (role `goal_summary`) to write short commentary from the already-computed results as JSON; the LLM never calculates or invents a number. The output guard in `BaseAgent.run()` may make one more call to reword, so at most two calls happen per request.
- **Inputs:** `state["goal_params"]` = `goal_amount`, `horizon_years`, `current_savings`, `monthly_contribution`, `risk_profile` (`low`/`medium`/`high`), optional `annual_step_up_pct` (default 0; a value above `goal_planning.max_step_up_pct` is **rejected** with `needs_input`, not capped), optional `inflation_adjust` (default `goal_planning.inflation_adjust_default`, currently `true`; the request overrides it). Validation (`src/core/goal_inputs.py`): finite numbers only (`inf`/`nan` are "not a number"; an int of any size is compared with the limits as an int, so a value such as `10**400` gives the `too_large` / horizon-range message instead of an overflow), `goal_amount` > 0, savings and contribution >= 0 (both may be 0: the result then shows what is needed), a whole-number horizon within `min_horizon_years`..`max_horizon_years`, a known risk profile. Any problem gives `needs_input` with every message listed and the LLM is not called. Risk profile is pre-filled from the profile store when saved (UI step, not built yet).
- **Currency:** `goal_planning.currency` (INR); amounts are never converted.
- **Math (pure functions, formulas in the docstrings):**
  - monthly rate from the annual **effective** rate: `(1 + annual/100)^(1/12) - 1`.
  - month-by-month projection, **deposits at the end of each month**: `balance = balance x (1 + monthly) + contribution`; with a step-up the contribution is multiplied by `(1 + step_up/100)` **every 12 months** (from month 13), so it is exact.
  - year-end table (year 0 to the horizon) per scenario (`conservative`, `expected`, `optimistic`); shortfall = target - final value (negative = surplus).
  - **required monthly contribution:** the final value is linear in the starting contribution, `FV(c) = FV(0) + c x FV_unit`, so `c = (target - FV(0)) / FV_unit` is exact for any rate and step-up (no search needed); with no step-up it equals the textbook annuity formula; 0 if savings alone reach the target.
  - **labels of the required amount** (`goal_planning.required_labels`, in the payload as `required_labels` and per scenario as `required_label`): "monthly amount needed to reach the goal"; with a step-up above 0 "starting monthly amount (rises X% a year)"; when the scenario already reaches the goal "minimum monthly amount to still reach the goal" (with the step-up wording appended when a step-up is set).
  - **months to goal** at the current contribution: first month k whose balance reaches the goal, 0 if already met (with inflation on the goal keeps rising, so month k counts as reached when `balance_k >= goal_today x (1 + i)^(k/12)`, not when it reaches the fixed target at the horizon; with inflation off the goal is a fixed amount), `None` with a note if not within `goal_planning.max_months`. This is the **first crossing only**: the rising goal uses `(1 + i)^(k/12)`, which equals the horizon target at month `horizon x 12`, but a balance can cross and later fall back below the price when its return is below inflation. Such scenarios are flagged `below_inflation` (inflation on and `annual_return_pct < inflation`; equal is not below) and `months_note` gets `goal_planning.below_inflation_months_note` appended.
  - **inflation** (when on): the entered goal is read as today's money, the **target** is the goal in future money (`goal x (1 + i)^years`), and the projection is also given in today's money (`value / (1 + i)^years`). Both figures are labelled. Shortfall and required contribution are measured at the horizon against that target; months to goal use the rising goal described above. A scenario can therefore reach the rising goal early yet show a shortfall at the horizon when its return is below inflation.
- **Assumptions (`goal_planning.assumptions` in config):** annual return per risk profile for the three scenarios, an inflation figure, each with `as_of` and a named `source`, labelled "illustrative, not predictions". Config load fails if a profile's returns decrease from conservative to optimistic, if any return or the inflation figure is negative, or if `max_months` does not cover `max_horizon_years` (`src/utils/config.py`).
- **Commentary step:** the prompt (`goal_planning.summary_prompt`) receives only `summary_input()` JSON with every number pre-formatted as text in **western digit grouping** (`1,234,567.00`), because the commentary should read the same everywhere; the shared number guard now also reads Indian grouping (`1,25,000` = 125000) and takes `1.25 lakh` / `2 crore` as 1.25 / 2 exactly as written, so western grouping is a choice, not a limit. Each scenario carries precomputed text: `surplus_or_shortfall_text` ("surplus of X" / "shortfall of X"), `years_months_to_goal` ("17 years 4 months", "already met by the current savings" or "not within the maximum period"), `time_compared_with_horizon`, and `monthly_amount_meaning` (the label above), and `status_text` (one sentence stating both facts: when the balance first covers the goal price, and the surplus or shortfall at the end of the horizon; six templates in `goal_planning.texts.status_*`, none says "reached", so a crossing followed by a shortfall is not contradictory); a `below_inflation` scenario also carries `below_inflation_text` ("At X% a year this scenario grows slower than the Y% rise in prices, so the savings buy less over time.") and the payload `warnings` get one entry per such scenario. When inflation is applied the input holds four plain sentences: the goal in today's money, the inflated goal, "the goal price at the end of the horizon" (used for the surplus or shortfall and the required amount) and "the goal price keeps rising with inflation" (used for the time to reach it); with inflation off none of them is present. Every number in these strings comes from the same computed values, so the number guard still passes (tested). The prompt asks for plain words (no quoting of field names or labels), one short paragraph (illustrative assumptions, not predictions, how the time to reach the goal compares with the horizon), one bullet per scenario (surplus or shortfall and the time to reach the goal at the current monthly amount), and one closing sentence naming in general terms what changes the outcome (monthly amount, time horizon, risk profile, yearly step-up) in observation wording; no product recommendations, no "you should", and numbers copied exactly (no lakh/crore or rounded forms). The shared number guard (`numbers_in` / `collect_numbers`) rejects commentary with any number not in the JSON, **except** that the `assumptions` entry (as_of date, timing notes, source) is shown to the LLM but is not whitelisted, so only computed values and the user's own inputs are allowed numbers (the prompt tells the LLM to refer to the notes in words); on an LLM error, an empty reply or an unknown number the fixed `summary_fallback` (no numbers) is used, status stays `ok` and a warning is added. The three warning sentences and the "not applied" text are in `goal_planning.texts`; the horizon is written with `years_months_text` ("1 year", "15 years").
- **Outputs:** `GoalPayload` (`src/core/models.py`): `inputs`, `target_amount`, `scenarios` (return, final value, shortfall, required contribution, months to goal, note, value in today's money when inflation is on), `projections` (flat `{scenario, year, value, value_todays_money}` rows for a line chart), `required_monthly_contribution`, `months_to_goal` and `shortfall` (each keyed by scenario), `inflation` (both goal figures and labels, `{}` when off), `assumptions` and `warnings`. The `assumptions` block (`illustrative`, `label`, `as_of`, `source`, currency, returns, inflation, **deposit timing: end of each month**, **step-up: every 12 months**, rate convention) is built from config in code, so it is in the payload whatever the LLM does.
- **Uses:** deterministic compound-growth calculator, LLM role `goal_summary` (own role so usage is counted per agent).

### 2.5 News Synthesizer
- Summarises recent Google News RSS headlines for a ticker (US or `.NS`) or a free search phrase. No new dependency (standard library `urllib` + `xml.etree`). Headlines are listed with publisher and time; the summary does not predict prices.
- **Input:** `state["news_params"]` = `{"ticker": "RELIANCE.NS"}` or `{"query": "Nifty 50"}`, optional `max_items` (an int is clamped to `1..news.max_items`; any other type uses `news.max_items` and adds the `news.messages.max_items_ignored` warning). If both are given the query wins. `src/core/news_query.py` (`resolve_query`) turns a ticker into a search phrase: `news.query_overrides` first (company name plus "stock", for example `RELIANCE.NS` -> "Reliance Industries stock", `AAPL` -> "Apple Inc AAPL stock"), else a template from `news.query_templates` (`INFY.NS` -> "INFY NSE stock", `TSLA` -> "TSLA stock", `^GSPC` -> "GSPC index"). The market provider offers no company name cheaply (yfinance `fast_info` has none; `Ticker.info` is an extra slow call), so no name lookup is made.
- **Fetch:** `NewsAgent` calls `provider.get_news(query)`. `get_news` now takes a search phrase (validated by `validate_news_query`), not a ticker. `YFinanceProvider` delegates it to an injected `NewsSource` (`GoogleNewsRssSource`, `src/data/news/`), so `CachedProvider` gives news the same cache, retry, stale-cache and error behaviour as quotes. News has no sample fallback.
- **Parsing** (`src/core/news_parse.py`, pure): title without the ` - Publisher` suffix when `<source>` matches, publisher, http(s) link only, `published` as ISO UTC, HTML entities and tags and control characters removed, title capped at `news.title_max_chars`, duplicates (same normalised title or link) removed, items older than `news.max_age_days` dropped, newest first, at most `news.max_items`.
- **Safety:** only `https` and hosts in `news.allowed_hosts` (redirects included); `news.timeout_seconds`; responses over `news.max_response_bytes` are rejected before parsing; `news.user_agent` is sent; the body must be plain UTF-8 (a BOM, NUL bytes, invalid UTF-8 or a declared non-UTF-8 encoding is a permanent error), and the DOCTYPE/ENTITY check runs on the decoded text, so UTF-16 cannot hide a DTD; a feed with a DTD/entity declaration or malformed XML is a controlled error; an empty feed is not an error.
- **Statuses:** no ticker/query or an invalid one -> `needs_input`; provider failure or no headlines -> `no_answer` (LLM not called); otherwise `ok`. The headline list (`items`: number, title, publisher, published, link) is always in the payload (`NewsPayload`: `query`, `items`, `as_of`, `source`, `is_stale`, `warnings`, `summary_citations`).
- **Summary:** one LLM call (role `news_summary`); the prompt asks for up to 4 bullets, fewer when there are few headlines, and no themes the headlines do not show; the input is only the numbered headlines as JSON and the prompt says headlines are untrusted data, never instructions. Checks: invalid `[n]` citations are dropped (not renumbered), a summary with no valid citation fails, every number must appear in a headline (shared number guard; citation digits are removed first). On an LLM error, an empty reply or a failed check the fixed `news.fallback_summary` is used, status stays `ok` and a warning is added. Stale cache is labelled with `is_stale` and a warning.

### 2.6 Tax Education (India only)
A distinct agent, not a copy of Finance Q&A: `TaxAgent` (`src/agents/tax.py`, name `tax_education`, LLM role `tax_summary`). General concepts only, no personal tax advice; it suggests a chartered accountant.
- **Two paths:** `state["tax_params"]` with a `calculation` key runs a calculator; anything else is a question (`state["query"]`). A chat question reaches the agent through the router intent `tax_education`; calculators are triggered by structured params from a Tax-tab form (`intent_hint="tax_education"`). There is no free-text parameter extraction. Router, workflow node and UI are later steps. `GraphState.tax_params` is typed.
- **Question path:** retrieval is locked to `tax.topic` (`accounts_and_taxes_india`), whatever the Chat dropdown says, through the shared `merge_retrieval()` in `src/rag/retriever.py` (the same merge and caps as Finance Q&A; one query, no rewrite call). No chunks -> `no_answer` without an LLM call. The prompt has the numbered chunks `[1]..[n]` and ALL config figures as numbered facts `F1..Fn` (label, value, unit, as_of, source, verify; the three equity figures also carry the tax-paid condition). The prompt rules: answer only from chunks and facts, cite chunks as `[n]` and facts as `[Fn]`, every number from a fact, chunk or the question, say "illustrative, verify before relying" with the as_of date when quoting a fact, never state an equity rate without the tax-paid assumption, no personal advice. Checks: the `NO_ANSWER` sentinel; invalid `[Fn]` dropped (not renumbered, `drop_invalid_fact_citations` in `src/core/citations.py`); chunk citations renumbered by `apply_citations`; at least one valid citation of either kind; every number must be in the retrieved chunks, the question or the `display` text of a figure (fact ids, as_of dates and source strings are not a whitelist, so the prompt tells the model not to write dates). A failed check gives `no_answer` with `tax.discarded_message`. A figure counts as PRESENT when the answer cites it (`[Fn]`, valid) OR states its number (a number in the answer that matches a number in the figure's `display` text; `quoted_fact_keys` in `src/core/tax_metrics.py`). For the union of the two, code (not the LLM) appends the fixed footer, lists the figures in `config_used`, and sets `exclusions` to the capital-gains list (with the tax-paid line if any present figure has `stt_condition`); `fact_citations` lists the cited facts only. So "Long-term gains are taxed at 12.5% [1]." gets the footer, the as_of dates, the sources in `config_used` and the tax-paid exclusion even though no fact was cited. A coincidental match (12.5 in two figures, 12 in "12 months") can add an extra footer or an extra `config_used` entry; this is accepted, because a missing footer on a stated rate is worse. An answer that states no figure number and cites no fact gets no footer and an empty `exclusions` list (the payload keeps the label). The footer holds: the label, the as_of dates of the cited figures and, for equity figures, the tax-paid assumption.
- **Calculators** (pure, `src/core/tax_metrics.py`, inputs validated in `src/core/tax_inputs.py`, formulas in the docstrings, `Decimal` maths rounded half-up to two places):
  - `capital_gains`: `asset_type` is `listed_equity`, `equity_fund`, `property_or_unlisted`, `other` or `debt_fund_post_2023`; inputs `buy_value`, `sell_value`, `expenses`, `holding_months`, optional `ltcg_exemption_already_used`, `slab_rate_pct`. Long-term means MORE than the configured holding months (equity or other threshold). Equity: short-term and long-term rates, yearly exemption (reduced by what was already used). Property, unlisted and other assets: long-term rate with no indexation and no exemption. Debt funds acquired after the rule change, and short-term property/other gains, are taxed at the slab rate the user enters (`needs_input` when missing; no rate in config). Cess is computed on the unrounded tax; the tax and the total are each rounded once and the shown cess is total minus tax. `cess_pct` is not listed for a loss. A loss or zero gain never needs `slab_rate_pct`. A gain of zero or less gives zero tax and an informational note only. Output: gain, term, exemption applied, taxable gain, tax, cess, total, step list.
  - `small_savings_growth`: scheme `ppf`, `nsc` or `ssy`, `mode` `annual` or `one_time`, `amount`, `years`; annual compounding at the configured rate, deposits at the start of each year, rate held constant, labelled approximate. Scheme limits and lock-ins are not checked.
- **Commentary:** one optional LLM call (`tax_summary`) on JSON of inputs, results, rates used and caveats, with every amount written as text with the currency sign (`tax.currency_symbol`) and western digit grouping (`₹15,500.00`), every rate with a percent sign (`12.5%`) and the rates used as their `display` text; the prompt says to copy them exactly (same idea as Goal Planning's `summary_input`). Every number in it must appear in the inputs, results or rates (dates and sources are not whitelisted). An LLM error, an empty reply or an invented number gives the fixed `tax.fallback_commentary` and a warning; status stays `ok`.
- **Config-driven figures:** every rate, limit and threshold lives in `config.yaml` under `tax.figures`; each has `label`, `value`, `source_short` (a short source text for outputs; never cut from the long `source`), `display` (the text the reader and the number guard see, e.g. `12.5%` or `₹1,25,000 (₹1.25 lakh) per financial year`), `unit`, `as_of`, a named `source` and `verify` (true until a human has checked it on incometax.gov.in). `_validate_tax` in `src/utils/config.py` rejects a figure without them. No figure appears in code, prompts, articles or this document; a test scans the tax modules for numeric literals other than 0, 1, 2 and 100 and the tax texts for digits. Debt funds acquired on or after the rule change have no config rate.
- **Exclusions** are per calculation (`tax.exclusions.capital_gains`, `tax.exclusions.small_savings`; `exclusions_for()` in `src/core/tax_metrics.py`) and are the only ones in that calculation's payload, commentary input and result. For capital gains: surcharge, indexation for older property purchases, loss set-off and income tax slabs are not modelled, and `tax.stt_assumption` comes first only for listed equity and equity funds (the equity rates apply only then), so a debt-fund result has no tax-paid line. For small savings: annual compounding, start-of-year deposits with a constant rate, and unchecked scheme limits and lock-ins (verify with the official scheme rules). The question path and a `needs_input` for an equity capital-gains request carry the equity capital-gains list.
- **Outputs:** `TaxPayload` (`path`, `calculation`, `inputs`, `results`, `steps`, `notes`, `assumptions`, `config_used` with as_of and source, `exclusions`, `label`, `facts`, `fact_citations`, `warnings`), the commentary or answer, and sources (cited chunks).

---

## 3. BaseAgent, AgentResult, GraphState contract

### 3.1 BaseAgent
`BaseAgent` is an **abstract parent class, not a seventh agent**. It is never a graph node.

**Pattern: Template Method.** `run()` fixes the shared steps; subclasses fill in `_execute()`.

```python
class AgentResult(TypedDict):
    agent: str
    turn_id: str                # id of the user turn that produced this result
    status: Literal["ok", "no_answer", "wrong_agent", "error", "needs_input"]
    answer: str                 # plain answer; NO disclaimer (added later, see below)
    sources: list[Source]
    error: str | None
    data: MarketPayload | PortfolioPayload | GoalPayload | TaxPayload | None  # typed per agent

class BaseAgent(ABC):
    name: str
    llm_role: str = "default"

    def __init__(self) -> None:
        self.llm = get_llm(role=self.llm_role)   # never construct an LLM directly
        self.log = get_logger(self.name)

    def run(self, state: GraphState) -> AgentResult:   # Template Method
        ...  # log start -> _execute -> log end; catch errors (no disclaimer here)

    @abstractmethod
    def _execute(self, state: GraphState) -> AgentResult: ...
```

`AgentResult` is a common envelope. Status meanings: `ok` (answered), `no_answer` (ran but found nothing, e.g. no relevant source), `wrong_agent` (query is not for this agent), `error` (exception caught), `needs_input` (a required input is missing, e.g. no ticker or empty portfolio). The payload in `data` is a per-agent typed dict (`MarketPayload`, `PortfolioPayload`, `GoalPayload`, `TaxPayload`, ...) that the UI uses for tables and charts.

Shared behaviour in `run()`:
- **Logging:** start, end, duration, and errors via `get_logger()`; never `print`.
- **Disclaimer (not here):** `run()` returns the plain answer. The disclaimer from `get_config()["app"]["disclaimer"]` is appended exactly once, by the final-answer nodes only: `synthesizer`, `safe_response` and `fallback`. Agent outputs are merged before the disclaimer is added, so a two-agent answer never shows it twice.
- **Output guard:** a rule-based phrase check on the answer before returning (section 3.3).
- **Error handling:** any exception becomes an `AgentResult` with `status="error"` and a friendly message. It is never re-raised into the graph.
- **Usage:** token counts are recorded by `UsageTracker` (section 10), not by the agent.

**Example:** `MarketAgent._execute()` only fetches quotes and builds the prompt; logging and error handling come free from `BaseAgent.run()`.

### 3.2 GraphState and reducers

```python
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # conversation memory
    query: str
    turn_id: str                  # new for every user turn (set by the facade)
    intent_hint: str | None       # set by Portfolio/Market/Goals tabs
    intents: list[str]            # from router, e.g. ["market", "finance_qa"]
    router_confidence: float
    agents_to_run: list[str]      # ordered queue (single writer: router)
    route_attempts: int           # re-route counter (section 4.4)
    agent_outputs: Annotated[list[AgentResult], operator.add]
    errors: Annotated[list[ErrorEntry], operator.add]      # {turn_id, message}
    sources: list[Source]         # single writer: synthesizer
    market_freshness: dict | None
    portfolio: list[dict] | None  # from Portfolio tab / profile store
    goal_params: dict | None      # from Goals tab
    market_params: dict | None    # from Market tab: {"tickers": [...], "period": "6mo"}
    topic_filter: str | None      # set by the Chat tab dropdown; the facade passes it to the retriever
    final_answer: str
```

**Why reducers:** in LangGraph a node's return value is merged into state key by key, and **without a reducer a node REPLACES the key**. If `market` returned `{"agent_outputs": [r1]}` and `finance_qa` later returned `{"agent_outputs": [r2]}`, only `r2` would survive. With `Annotated[list[...], operator.add]` the returned list is appended instead.
- **Each node returns only its own result in a list**, e.g. `{"agent_outputs": [result]}`; it never reads and re-returns the whole list.
- **A list, not a dict keyed by agent name,** because it keeps execution order and keeps both results if an agent runs twice (for example after re-routing).
- **Single-writer fields** (e.g. `agents_to_run`, `route_attempts`, `final_answer`) need no reducer: one node sets them, so replacing is the intent.
- **Token totals are not in state.** They come from `UsageTracker` (section 10).

**Per-turn state hygiene.** The checkpointer persists state per `thread_id`, so without care the previous turn's values carry over into the next turn: the append-reducer keys (`agent_outputs`, `errors`) would keep growing, and the single-writer keys (`intent_hint`, `intents`, `agents_to_run`, `route_attempts`, `final_answer`) would keep their old values. Rules:
- **`facade.run_query` passes a complete, fresh input on every call:** it resets `intent_hint` (to the tab's hint or `None`), `intents`, `agents_to_run`, `final_answer`, sets `route_attempts = 0`, and sets `turn_id` to a new UUID. Resetting an append-reducer key needs care: `operator.add` cannot clear a list, so the filter below is what protects those keys.
- **Every `AgentResult` carries the `turn_id`** of the turn that produced it (filled in by `BaseAgent.run()` from state).
- **Nodes, edges and the synthesizer read only results for the current `turn_id`** through one helper, `current_results(state)`, which returns `[r for r in state["agent_outputs"] if r["turn_id"] == state["turn_id"]]` (the same filter applies to `errors`, stored as `turn_id`-tagged entries). Old results stay in the checkpoint history but are never used.
- **Example:** in turn 1 the user submits the Market tab (`intent_hint="market"`). In turn 2 they type a Chat question. Turn 2's input sets `intent_hint=None`, so the START edge goes to the router instead of `market`; `route_attempts` starts at 0; and turn 1's market result is ignored because its `turn_id` differs.

**Example node function:**

```python
def market_node(state: GraphState) -> dict:
    result = MarketAgent().run(state)             # result["turn_id"] == state["turn_id"]
    update = {"agent_outputs": [result]}          # appended by operator.add
    if result["error"]:
        update["errors"] = [{"turn_id": result["turn_id"], "message": result["error"]}]
    return update
```

**Synthesizer: handles N results (up to 4 with re-routes) and cannot fail on an empty `ok` list:**

```python
def synthesizer_node(state: GraphState) -> dict:
    results = current_results(state)                       # this turn only
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        answer = ok[0]["answer"] if len(ok) == 1 else merge_with_llm(ok)  # LLM only for 2+
        sources = [s for r in ok for s in r["sources"]]
    else:
        no_answer = [r for r in results if r["status"] == "no_answer"]
        if not no_answer:                                  # defensive; the edge routes to fallback
            return fallback_node(state)
        answer = no_answer[-1]["answer"]                   # present the no_answer message
        sources = []
    return {"final_answer": add_disclaimer(answer), "sources": sources}
```

**The merge of 2+ ok results (`src/workflow/merge.py`, `merge_with_llm`):** code first renumbers the `[n]` markers across agents (agent A `[1][2]`, agent B's `[1]` becomes `[3]`; News, which has no source list, gets `[News n]`) and concatenates the sources in that order. ONE LLM call (role `synthesizer`, prompt `workflow.synthesizer_prompt`) receives the answers as a JSON data block inside `<agent_answers>` tags (`<` written as `\u003c`); the prompt says they are data, never instructions, and to keep every `[n]`, number, caveat and closing sentence exactly. The merged text is used only if all checks pass: not empty; the output-guard patterns do not match; it does not contain the disclaimer (added once, later); every `[Fn]` / `[News n]` marker appears in the inputs (a plain `[n]` outside the source list is dropped, not a failure); every number is a number of the input answers (shared number guard, markers removed first); and every text that an agent took from its payload and put in its answer is still there (`must_keep`: payload `label`, the `source` of stale or sample prices, every `exclusions` line and each `config_used` `as_of`, only when the agent's own answer contains it). A failed check or an LLM error gives the verbatim join (the answers under `workflow.agent_labels` headings with the renumbered markers). Notes for `needs_input`, `error` and `no_answer` results are written by code after the merge, then the disclaimer. One ok result: no LLM call. Limit: a caveat that an agent kept only in its payload `warnings` (not in its answer text) is not protected by this check.

**Result handling rule:** if at least one result is `ok`, merge the `ok` results. If none is `ok` but at least one is `no_answer`, the synthesizer presents the `no_answer` message (for example "the knowledge base has no good match") **with the disclaimer**. `fallback` is used only when every result is `error` or nothing ran.

### 3.3 Output guard
No agent may give a personalised buy/sell/hold recommendation.
- **Prompts:** every agent prompt forbids personalised buy/sell/hold advice.
- **Why a second layer:** tab requests skip the router, so the `specific_advice` guardrail (section 4.3) never sees them. A one-stock portfolio can tempt the commentary to say "you should sell".
- **Rule-based check:** `BaseAgent.run()` runs a cheap phrase check on the answer before returning it (patterns such as "you should (buy|sell|hold)", "I recommend (buying|selling)", "sell your", "buy more"; list in config). On a match it replaces the answer with a short educational rewording request (one retry) or, if still matching, a safe message, and logs a warning. Because it lives in `run()` (Template Method), every agent gets it for free.
- **[Later]:** an LLM-based check for subtler phrasing.

---

## 4. LangGraph design

### 4.1 Nodes

| Node | Role |
|---|---|
| `router` | LLM role `router`; classifies query into intents with a confidence value (structured output) |
| `finance_qa`, `portfolio`, `market`, `goal_planning`, `news`, `tax_education` | wrap one agent each |
| `clarify` | returns a clarifying question as an ordinary assistant reply (section 4.5) |
| `safe_response` | `out_of_scope`: static text, no LLM. `specific_advice`: `merge_retrieval` for context, then an LLM (role `safe_response`) explains how people weigh such a decision, citing `[n]`, checked in code; static text on any failure; appends the disclaimer |
| `synthesizer` | one ok result: its answer, no LLM call; 2+ ok results: one checked LLM merge with a verbatim fallback (section 3.2); code-written notes; appends the disclaimer |
| `fallback` | graceful message when routing fails or all agents fail; appends the disclaimer |
| `reroute` | internal bookkeeping node (not an agent): counts one re-route hop and queues the target agent (section 4.4) |

### 4.2 Entry: every request goes through LangGraph
A conditional edge at `START` checks `state["intent_hint"]`:
- **Hint present** (Portfolio, Market and Goals tab submissions): go straight to that agent node. No router call, so no router tokens.
- **No hint** (Chat tab): go to `router`.

### 4.3 Router logic and guardrail
Intents: `finance_qa`, `portfolio`, `market`, `goal_planning`, `news`, `tax_education`, `clarify`, `out_of_scope`, `specific_advice`. The router also returns `confidence` (0 to 1).
- `specific_advice` catches requests like "Should I buy X?" or "Which stock will go up?".
- `out_of_scope` catches non-finance topics.
- **No mixed routing:** if `specific_advice` is present, alone or alongside other intents, the whole query goes only to `safe_response`. No other agent runs.
- `safe_response` gives educational framing ("here is how people think about evaluating a stock...") and makes no recommendation. For `specific_advice` (`src/workflow/advice.py`): `merge_retrieval` on the query (and the Chat topic filter); no chunks means the static `workflow.messages.advice_static` text without an LLM call. Otherwise one LLM call (`workflow.safe_response_prompt`; the chunks in a `<context>` block, the query as JSON in `<user_input>`; the prompt forbids any recommendation, ranking or prediction and allows only the context). Code then requires at least one valid `[n]` (renumbered by `apply_citations`), no number outside the chunks and the question, and runs `guard_answer` (reword once, then the safe message). A reworded text must pass the number check again; a replaced or reworded text carries no sources. Any failure gives the static advice text. `out_of_scope` is always static. No agent runs in this node.
- **Low confidence** (below `workflow.router_min_confidence` in config) goes to `clarify`.
- **Low-confidence `out_of_scope`:** `out_of_scope` with no agent intent goes to `safe_response` (a refusal) whatever the confidence, never to `clarify`; only a query that still has an agent intent, or the `clarify` intent, is sent to `clarify` for low confidence.
- **Priority** (edge `route_after_router`): `specific_advice` anywhere, then `out_of_scope` when no agent intent is left, then `clarify` (intent or low confidence), then the first agent in `agents_to_run`, else `fallback`. `out_of_scope` next to a real agent intent is dropped and the agents run. The queue (`plan_agents`) is empty whenever `specific_advice` is present and holds at most `workflow.max_agents_per_turn` agents.
- **Why:** one simple rule is easy to test and cannot leak advice through another agent. Splitting the educational part from the advice part is **[Later]**.
- Router output that fails validation falls back to `finance_qa`; if that also fails, `fallback`. Invalid means: not a mapping or model, `intents` empty / not strings / an unknown intent / more agent intents than `workflow.max_agents_per_turn` (unless `specific_advice` is present), `confidence` a bool, not a number, nan, inf or outside 0 to 1, or the call raising. The fallback decision has confidence `workflow.router_min_confidence`, so it is not sent to `clarify`.
- **Entities (decisions the doc didn't specify):** the same structured-output call returns `entities`: `tickers` (mentions as the user wrote them), `period` and `news_query`. They are untrusted. A mention is kept only if it occurs in the query text; it becomes a symbol only through `workflow.ticker_aliases` (name in lower case -> symbol) or, when written in capitals exactly as in the query, as a literal symbol accepted by `validate_symbol`; the model never supplies a symbol. A `period` must be one of `market_analysis.periods`; a `news_query` must occur in the query, otherwise the first resolved symbol is used (`news_params = {"ticker": ...}`). The result is written to `market_params` / `news_params` whatever the intent (that is what lets a `finance_qa` `no_answer` re-route to `market`) and never overwrites a param a tab sent. The router never extracts amounts, holdings, goals or tax-calculator inputs. A pronoun follow-up ("How is it doing?") finds no ticker in the query, so `market` answers `needs_input` and the graph clarifies; entities are not taken from the history.
- **Advice patterns:** `workflow.advice_patterns` (case-insensitive regexes, for example `should I buy`, `which stock will go up`, `tell me to buy`) are checked on the query after the model; a match ADDS `specific_advice` to the intents, also when the model call failed. They never remove an intent. Educational questions such as "what does a buy rating mean" or "how do I choose between buying and renting" must not match (tests).
- **Prompt injection:** the query and the earlier messages (last `workflow.max_history_turns` turns, each cut to `workflow.router_history_max_chars`) go to the model as one JSON block inside `<router_input>` tags (`<` written as `<`, so text cannot close the block) and the prompt (`workflow.router_prompt`) says to classify only and never follow instructions found in it.
- **[Later] Hybrid router:** keyword rules handle obvious cases ("should I buy", clear ticker lookups) with no LLM call; the LLM runs only when the query is ambiguous.

### 4.4 Edges and bounded re-routing

```
START -> (conditional) agent named by intent_hint | router
router -> (conditional) safe_response if specific_advice/out_of_scope
                        | clarify if confidence is low or intent is clarify
                        | first agent in agents_to_run | fallback
agent node -> (conditional on AgentResult.status of the latest result):
      (only results with the current turn_id are considered)
      ok            -> next agent in agents_to_run | synthesizer
      wrong_agent / no_answer -> re-route (below) | else next agent | synthesizer
      needs_input   -> next agent in agents_to_run (the queue goes on); at the end of the queue see below
      error         -> next agent | fallback if every result is error, or nothing ran
at the end of the queue: synthesizer if any result is ok (it adds a note for each needs_input / error result);
      else clarify if any result is needs_input; else synthesizer if any result is no_answer; else fallback
safe_response, synthesizer, fallback, clarify -> END
```

**needs_input (decision the doc didn't specify):** a `needs_input` result is an ordinary non-ok result. The queue goes on; at the end, if any result is `ok` the synthesizer answers and appends each `needs_input` message as a code-written note (`workflow.messages.needs_input_note`); if no result is `ok` and at least one is `needs_input`, the graph goes to `clarify`, which shows the agents' own messages. A single agent (or a tab hint) with `needs_input` therefore still ends in `clarify`.

**Re-routing example:** `finance_qa` returns `no_answer` for "what is NVDA doing today", but the query contains a ticker, so the conditional edge sends it to `market`. `route_attempts` is incremented on each re-route and **capped at 2 hops**; at the cap the graph goes to `synthesizer` or `fallback`. LangGraph's recursion limit (set in config) is only a backstop.

**All-agents-failed path:** if every result is `error`, or nothing ran, the edge goes to `fallback`, not `synthesizer`. If no result is `ok` but at least one is `no_answer`, the edge goes to `synthesizer`, which presents the `no_answer` message with the disclaimer (section 3.2). A user who asked something the knowledge base cannot answer gets that honest message, not a generic failure.

**Disclaimer:** added once, by `synthesizer`, `safe_response` and `fallback`. `clarify` asks a question and gives no financial content, so it adds none.

### 4.5 Clarifying questions
Triggered by low router confidence or an agent returning `needs_input` (e.g. "How is it doing?" with no ticker).
- **[MVP]** the `clarify` node returns the question as an ordinary assistant message and the graph ends. The next user message continues in the same `thread_id`, and the router sees the question in `messages`.
- **[Later] stretch:** true human-in-the-loop with LangGraph `interrupt()` and resume with `Command(resume=...)`. It requires a checkpointer (section 12). The API must be checked against current LangGraph docs when built.

### 4.6 Two-agent queries
Example: "What is Apple trading at, and what is a P/E ratio?" gives intents `["market", "finance_qa"]`.
- **[MVP] sequential:** router sets `agents_to_run=["market","finance_qa"]`; a conditional edge picks the next agent after each node; each node appends its result to `agent_outputs` (reducer); `synthesizer` merges them with one `default`-role call.
- **[Later] parallel:** replace the pick-next edge with LangGraph `Send` fan-out. State schema is unchanged because of the reducer.
- If one agent errors, the other's output is still returned with a note (partial result).

### 4.7 Short-term memory
`messages` lives in graph state and is persisted per `thread_id` by the checkpointer (section 12). History is trimmed to `workflow.max_history_turns` before each LLM call.

---

## 5. RAG

**Knowledge base:** 50-100 original educational articles written for this project (Markdown in `src/data/knowledge_base/`), each citing named reference sources (e.g. SEC Investor.gov, FINRA, and for India incometax.gov.in, SEBI). No copied text. Front matter: `title`, `topic`, `references`.

**Topics:** stocks; bonds; ETFs and funds; risk and diversification; goals and retirement; accounts and taxes (India); market basics; glossary. General content is jurisdiction-neutral; tax content is India only and uses topic `accounts_and_taxes_india`.

### 5.1 Offline: `rag/ingest.py`

| Step | Decision | Why |
|---|---|---|
| Load | Markdown loader reads front matter into metadata | Keeps citation data with the text |
| Chunk | `RecursiveCharacterTextSplitter`, `rag.chunk_size`, `rag.chunk_overlap` (tuned values 300 and 120, section 5.3) | Paragraph-sized chunks; overlap keeps context across boundaries |
| Embed | Local `sentence-transformers/all-MiniLM-L6-v2` (`embeddings.*`) | Free, no API budget, runs offline |
| Build and save | FAISS index in `rag.vector_store_dir` plus an **index manifest** | Fast, simple, no server |

**Manifest** (`manifest.json` beside the index): embedding model name, chunk size and overlap, and a hash of the article set and the per-topic overrides. It is checked when the index loads; a mismatch with current config logs a warning (the index may be stale and should be rebuilt).

**Per-topic chunking:** `rag.topic_chunking` can override `chunk_size` and `chunk_overlap` for one topic; `glossary` starts at 250 and 0. Glossary entries are tiny, independent definitions, so one entry per chunk retrieves better than a blend of several unrelated terms. The overrides are recorded in the manifest (a change triggers the stale-index warning), and the eval sweep varies only the default settings while glossary articles keep their override.

### 5.2 Online: `rag/retriever.py`
1. Embed the query with the same model as ingest.
2. Search **all** chunks (an exact flat index, cheap at this size; revisit with over-fetching, i.e. asking for more than `top_k` and trimming, if the knowledge base grows to tens of thousands of chunks). The optional `topic` filter (used by Tax Education) is applied to these hits **before** selection.
3. Pass the hits to the shared pure function `select_chunks()`, which applies the score threshold, sorts best first, enforces the diversity rules and returns at most `rag.top_k` chunks (6 after tuning, with `rag.score_threshold` 0.30). The rules are `rag.max_chunks_per_article` (default 2) and `rag.max_chunks_per_topic` (per-topic caps; `glossary` is capped at 1, because several tiny definitions often match at once and crowd out the real explanation). A cap that is empty or null is off.
4. Build the prompt: use only the provided context, cite `[1]..[n]`.
5. Answer with citations.

The eval harness (section 5.3) calls the same retrieval path as the app, `merge_retrieval()` over `retrieve()` (acronym expansion, written query first, `select_chunks()`), through `production_chunks()` in `tests/rag_eval/rag_eval.py`; only the store, the embedder and the rag settings are the eval's own, so it measures what the app does.

**Topic filter in the Chat tab:** the Chat tab offers an optional topic filter dropdown (default "All topics"). The selected topic is passed to the retriever as the `topic` metadata filter from step 2. Tax Education always uses its own locked `accounts_and_taxes_india` filter, whatever the dropdown says.

**Source attribution:** each chunk carries `{title, topic, references, chunk_id, score}`, returned in `AgentResult.sources`; the UI shows a "Sources" expander under each answer.
**No relevant source:** if no chunk passes the threshold, the agent returns `status="no_answer"` saying the knowledge base has no good match instead of guessing. This status also feeds re-routing (section 4.4).

### 5.3 Tuning and evaluation harness
`chunk_size`, `chunk_overlap`, `top_k` and the score threshold started as guesses (800, 120, 4 and 0.35) and were tuned with an evaluation harness (`tests/rag_eval/`). **Current values: `chunk_size` 300, `chunk_overlap` 120, `top_k` 6, `score_threshold` 0.30** (glossary keeps its 250/0 override). The sweep results are in the README, section "RAG tuning results". The harness works like this:
- **Question set:** about 30 labelled questions: question plus the article(s) that should answer it, including a few off-topic questions that should match nothing.
- **Sweep:** over `chunk_size`, overlap, `top_k` and threshold.
- **Metrics:** hit-rate@k (does a labelled article appear in the top k) and MRR (mean reciprocal rank of the first correct article).
- **Diversity on/off:** every sweep row is scored twice, with the `select_chunks()` caps on and off (a `diversity` column), using the app's retrieval path with the caps switched off in the rag settings it passes, so the table shows what the caps are worth.
- **Evaluation sets:** development questions, a held-out set (independent questions, skipped until labelled; its columns appear once at least one item is labelled) and an off-topic set made of `off_topic` plus `near_miss` questions, which should match nothing (false matches are counted separately as `off_topic_false` and `near_miss_false`).
- **Best row and diagnostics:** the "best row" of the sweep ranks rows with no false matches first, then by `hit_rate`, then by `mrr`. A diagnostics report (development misses and the score gap between answerable and off-topic questions) is printed for the `config.yaml` settings and for the best row.
- **Cost:** retrieval metrics use local embeddings, so the sweep costs no API tokens.
- **Threshold:** set from the score gap between answerable and off-topic questions (place it between the two score distributions).
- **Optional:** an LLM-judge faithfulness check on a small sample (uses tokens, so run it rarely and never in the normal test run).

### 5.4 Constraints and notes
- `all-MiniLM-L6-v2` truncates input at about 256 tokens, so `chunk_size` (in characters) must stay well under about 1000; text beyond that is silently ignored by the embedder.
- **Vector store behind an interface (Strategy):** `VectorStore` with `add()`, `search()`, `save()`, `load()`. `FaissStore` is the MVP choice; Chroma or Pinecone can be swapped in via `rag.vector_store` in config.
- **Temperature is not a RAG setting.** It belongs to the LLM role (section 9), and some models (such as gpt-6-luna) accept only their default temperature.
- **[Later]:** hybrid search or reranking.

---

## 6. Market data layer and caching

**Pattern: Strategy.** `MarketDataProvider` interface; `YFinanceProvider` is the **[MVP]** implementation. An `alpha_vantage` provider (named in config) is **[Later]**.

```
get_quote(ticker) -> MarketResult(data, currency, as_of, source, is_stale)
get_history(ticker, period) -> MarketResult
get_news(query) -> MarketResult      # query = search phrase; the News agent turns tickers into phrases
```

**Fallback chain (Chain of Responsibility):** `live -> retry -> stale cache -> sample`. Each handler either returns a result or passes the request on.
1. **Live fetch** through the cache; a fresh cache hit returns immediately.
2. **Retry:** `market_data.max_retries` attempts, exponential backoff from `market_data.backoff_seconds`.
3. **Stale cache:** on failure, return the expired entry with `is_stale=True`.
4. **Sample:** bundled snapshot from `src/data/` (US and `.NS` examples) with `source="sample data"`. Always labelled.
5. **Total failure:** `MarketResult` with error; the agent explains data is unavailable.

**News source (Strategy inside the provider):** `YFinanceProvider` takes a `NewsSource`; the default is `GoogleNewsRssSource` (Google News RSS search feed, `hl=en-IN gl=IN ceid=IN:en`, standard library only). Google RSS replaces yfinance news because the yfinance news path returned nothing usable for this project (reported by the project author, not re-verified in the commit that removed it), and the RSS feed also supports free search phrases such as "Nifty 50", which need no ticker. Headlines are never sampled: a news request that cannot be served live or from cache is an error result.

**Freshness indicator:** every `MarketResult` carries `as_of` (UTC), `source` (`live`, `cache`, `stale cache`, `sample data`) and the quote `currency`. The UI shows a badge: green "Live, updated 2 min ago", amber "Cached/stale", grey "Sample data".

### 6.1 Caching strategy

| What | Mechanism | TTL / lifetime |
|---|---|---|
| Market quotes | `CachedProvider` (Decorator) around the provider; in-memory TTL cache keyed by `(ticker, kind)` | `market_data.cache_ttl_minutes` (30) |
| Historical data | Same decorator, separate key kind | Longer TTL, `market_data.history_ttl_minutes` |
| Embedding model and FAISS index | Singleton via `st.cache_resource` (and a cached getter outside Streamlit) | Process lifetime |
| LLM calls and router decisions | Exact-match cache keyed by `(role, model, prompt)` | `llm.cache_ttl_minutes`; **market-dependent answers are never cached beyond the market TTL** |
| Semantic cache (similar prompts) | **[Later]** | n/a |

- **`CachedProvider` is transparent:** it implements the same `MarketDataProvider` interface, so an agent cannot tell whether it talks to the raw or the cached provider.
- **Rule: one caching layer per kind of data.** `st.cache_data` and our TTL cache never stack on the same data; market data uses only our TTL cache (it needs `as_of` and stale handling), models and indexes use only `st.cache_resource`.

---

## 7. Error handling and fallback strategy

| Layer | Failure | Response |
|---|---|---|
| Tool: market data | network error, rate limit, bad ticker | retry, then stale cache, then sample data, then error result |
| Tool: RAG | missing index, empty results | `no_answer` with "knowledge base unavailable / no match" message |
| Tool: tax calculators | missing `tax:` config value | `error` status naming the missing key |
| LLM call | timeout, rate limit | `max_retries` from config (set in `get_llm`); then agent error result |
| Agent | any exception | `BaseAgent.run()` returns `AgentResult(status="error")`, never raises |
| Agent | wrong agent, no answer | bounded re-route (max 2 hops, section 4.4) |
| Agent | missing input | `needs_input`, then `clarify` |
| Router | invalid or empty output | default to `finance_qa`, then `fallback` node |
| Graph | one of two agents fails | return partial result plus note |
| Graph | all fail | `fallback` node: friendly message, suggest retrying, disclaimer |
| Profile store | SQLite unavailable | app runs without saved profile; warning logged |
| UI | any uncaught error | `st.error` with friendly text; details only in logs |
| Input | invalid ticker, negative amounts | validated before the graph runs; inline form messages |

Principles: fail soft, always label degraded data, log the technical detail, show the user a plain message, never show stack traces.

---

## 8. Design patterns

| Pattern | Where | Concrete example |
|---|---|---|
| Factory | `core/llm_factory.py` | `get_llm(role="router")` returns a chat model; no agent names a provider |
| Template Method | `BaseAgent.run()` | `run()` adds logging and error handling around each agent's `_execute()` |
| Strategy | market providers, vector store | swap `YFinanceProvider` for `AlphaVantageProvider`, or `FaissStore` for Chroma, via config |
| Singleton (cached getter) | config, FAISS store, embeddings model | `get_vector_store()` loads the index once per process |
| Registry | agent registry in `workflow/` | router intents map to agent classes via a dict, so adding `news` is one entry |
| Observer (callback) | token tracker | a LangChain callback handler records usage after every LLM call |
| Facade | `workflow/run_query()` | the UI calls one function; graph details stay hidden |
| State (graph) | LangGraph `StateGraph` | `GraphState` carries all data between nodes |
| Repository | `ProfileRepository` | `repo.save_profile(name, risk="medium")`; the Goals agent never writes SQL |
| Decorator | `CachedProvider` | wraps `YFinanceProvider` and adds a TTL cache without changing its interface |
| Chain of Responsibility | market fallback chain | live fetch fails, so retry, then stale cache, then sample snapshot |

---

## 9. Model strategy

Configured per role, selected through `get_llm(role=...)`. **[MVP]**

```yaml
llm:
  provider: openai
  model: <default-model-id>        # default role
  temperature: 0.2                 # null if the model only supports its default
  max_retries: 3
  max_requests_per_minute: null    # null = no limit
  roles:
    router:
      model: <default-model-id>     # same as default for now; a cheaper ID can be swapped in later
    synthesizer:                    # optional; falls back to default
      model: <default-model-id>
    portfolio_summary:              # Portfolio agent summary (section 2.2)
      model: <default-model-id>
    market_summary:                 # Market agent summary (section 2.3)
    goal_summary:                   # Goal Planning commentary (section 2.4)
      model: <default-model-id>
    tax_summary:                    # Tax Education answers and calculator commentary (section 2.6)
      model: <default-model-id>

portfolio:                          # starting values, labelled "illustrative"
  max_holdings: 50
  max_workers: 8                    # concurrent quote fetches
  full_diversification_holdings: 20
  single_class_penalty: 15
  concentration_warning: 0.40
  equity_like_classes: [equity, other]
  diversification_bands: {concentrated_below: 40, diversified_from: 70}
  risk:
    equity_moderate_from: 0.40
    equity_high_from: 0.70
    concentration_moderate_from: 0.30
    concentration_high_from: 0.50
  needs_input_message: ...          # fixed texts and the summary prompt live here too (section 2.2)
  unpriced_message: ...
  summary_fallback: ...
  summary_prompt: ...               # contains the {metrics_json} placeholder

market_analysis:                    # starting values, labelled "illustrative" (section 2.3)
  max_tickers: 4
  max_workers: 4                    # concurrent history fetches
  default_period: 6mo
  periods: [1mo, 3mo, 6mo, 1y]
  period_days: {1mo: 30, 3mo: 91, 6mo: 182, 1y: 365}
  period_labels: {1mo: 1 month, 3mo: 3 months, 6mo: 6 months, 1y: 1 year}
  fetch_period: 1y                  # one history call per ticker, sliced in code
  sma_windows: [20, 50]             # trading days, short first
  trading_days: 252                 # to annualise volatility
  week52_min_days: 350
  short_history_tolerance_pct: 15   # history covering this much less than the period is flagged as short
  trend: {return_threshold_pct: 3.0, sma_gap_pct: 0.5}
  needs_input_message: ...          # fixed texts and the summary prompt live here too (section 2.3)
  unavailable_message: ...
  summary_fallback: ...
  summary_prompt: ...

goal_planning:                      # illustrative, not predictions (section 2.4)
  currency: INR
  inflation_adjust_default: true
  min_horizon_years: 1
  max_horizon_years: 40
  max_months: 600                   # must cover max_horizon_years
  max_step_up_pct: 15
  max_amount: 1000000000000
  risk_profiles: [low, medium, high]
  scenarios: [conservative, expected, optimistic]     # lowest return first
  assumptions:                      # label, as_of, source, returns_pct per profile, inflation {annual_pct, as_of, source}, timing notes
  labels / errors / needs_input_message / not_reached_note / already_met_note / summary_fallback / summary_prompt: ...

tax:                                # illustrative, every figure must be verified (section 2.6)
  topic: accounts_and_taxes_india
  label: illustrative, verify before relying
  figures:                          # the ONLY place a rate, limit or threshold may appear
    <figure_key>: {label, value, unit, as_of, source, verify, stt_condition?}
  units / asset_types / equity_asset_types / slab_asset_types / schemes: ...
  input_limits: {max_amount, max_months, max_years, max_pct}     # our own sanity caps, not tax figures
  stt_assumption / exclusions {capital_gains, small_savings} / errors / texts: ...
  system_prompt / fact_line / fact_condition / commentary_prompt / footer / fallback_commentary / no_answer_message / discarded_message: ...
```

- `get_llm(role="default")` reads `llm.roles.<role>` and merges over `llm.*`; unknown or missing role falls back to the default.
- **Temperature is optional per role.** Set `temperature: null` (or omit it) for models that only support their default, such as gpt-6-luna. `get_llm()` then does not pass it.
- **Router:** same model as default for now (decision), with structured output. It is skipped for tab submissions that carry an `intent_hint`. A cheaper model or the hybrid router (section 4.3) is the cost lever later.
- **Agents and synthesizer:** default model, since answer quality matters here.
- The existing `get_llm(temperature=None)` signature stays compatible: `role` is a new optional parameter.
- **Rate limiting.** A per-minute request limit is read from config (`llm.max_requests_per_minute`) and enforced in `get_llm`'s call path, so no agent can bypass it. It works alongside `max_retries`: the limiter keeps the app under the provider's limit in the first place, and retries handle failures that still happen. When the limit is reached, a call waits for a free slot instead of failing. If the key is missing or null, no limit is applied.
- **[Later]:** per-role provider.

---

## 10. Observability and token tracking

Three layers:

| Layer | Purpose | Where |
|---|---|---|
| `get_logger()` | text logs: node start/end, duration, errors, re-routes | all modules |
| `UsageTracker` | tokens and cost | `core/usage_tracker.py` |
| LangSmith (optional) | trace of every graph run, node and LLM call | off by default |

### 10.1 UsageTracker
- **How [MVP]:** a `BaseCallbackHandler` attached in `get_llm()`. On `on_llm_end` it reads `usage_metadata` and adds `input_tokens` / `output_tokens` to per-role, per-agent totals. It is the single source of token totals (not graph state).
- **Cost:** `tokens x price` using a `pricing:` table in `config.yaml` (per 1M tokens, by model ID). Unknown model gives tokens only, cost "n/a".
- **Where it surfaces:** session totals in the Streamlit sidebar; one log line per request.
- **Budget guard [MVP]:** optional `llm.max_session_tokens`; when exceeded, the UI warns and stops calling the LLM.
- **Why a callback:** tracking stays out of agent code, so no agent can forget to record usage.

### 10.2 LangSmith tracing
- Configured **only through environment variables**: `LANGSMITH_TRACING`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT` (read with `get_secret()`), plus `observability.langsmith.enabled` in `config.yaml`, which **defaults to `false`**. Tracing runs only when both the flag and the env vars say so.
- Runs are tagged with `thread_id`, agent name and LLM role.
- **Privacy warning:** traces send prompts and responses to the LangSmith cloud. Keep it off whenever real portfolio data is in use.
- The sidebar shows a status line: "LangSmith: off" or "LangSmith: on, project X".
- The three variables are **already in place** in `.env.example` (nothing to add in a build step). `.env` itself is never read or committed.
- **[Later]:** LangSmith datasets for the RAG and router test sets.

---

## 11. Modules and build order

### 11.1 Module list (matches existing folders)

| Folder | Modules |
|---|---|
| `src/utils/` | `config.py`, `logger.py` (exist) |
| `src/core/` | `llm_factory.py` (exists; add `role`), `usage_tracker.py`, `models.py` (AgentResult, payloads, Source, MarketResult), `portfolio_models.py`, `portfolio_metrics.py`, `portfolio_valuation.py`, `market_models.py`, `market_metrics.py`, `market_history.py`, `goal_models.py`, `goal_inputs.py`, `goal_metrics.py`, `profile_repository.py`, `tracing.py` |
| `src/data/` | `knowledge_base/*.md`, `sample_portfolios/` (US and `.NS`), `market_snapshot.json` (US and `.NS`) |
| `src/rag/` | `ingest.py`, `embeddings.py`, `retriever.py`, `vector_store.py` (interface and FAISS), `vector_store/` |
| `src/agents/` | `base_agent.py`, `finance_qa.py`, `portfolio.py`, `market.py`, `goal_planning.py`, `news.py`, `tax.py` |
| `src/data/market/` | `provider.py`, `yfinance_provider.py`, `cache.py` (`CachedProvider`) |
| `src/workflow/` | `state.py`, `router.py`, `nodes.py`, `graph.py`, `facade.py` |
| `src/web_app/` | `app.py`, `chat_tab.py`, `portfolio_tab.py`, `market_tab.py`, `goals_tab.py`, `components.py` (badges, Plotly charts) |
| `tests/` | mirrors `src/` layout; `tests/rag_eval/` for the harness |

### 11.2 Build order
Each step ships with tests and is small enough to review on its own.

1. **Core:** role-aware `get_llm`, `models.py` (envelope and payloads), `usage_tracker.py`.
2. **BaseAgent** with logging and error wrapper.
3. **Market data layer:** provider, `CachedProvider`, retry, fallback chain, snapshot (US and `.NS`).
4. **RAG:** write articles, ingest with manifest, retriever, evaluation harness, tune values.
5. **Agents (build first):** Finance Q&A, Market, Portfolio, Goal Planning.
6. **Workflow:** state with reducers, router with confidence and guardrail, nodes, graph (START edge, re-routing, clarify), facade.
7. **Persistence:** SQLite checkpointer, `ProfileRepository` (after dependency approval; `.gitignore` already covers `*.db`).
8. **Observability:** LangSmith flag, tags, sidebar status (`.env.example` already has the variables).
9. **Streamlit UI:** Chat first, then Market, Portfolio, Goals; sidebar.
10. **Agents (build after):** News Synthesizer, then Tax Education with `tax:` config.
11. **Polish:** README, docs, coverage gaps, demo script, tax-rate verification.
12. **Later:** CSV portfolio upload, parallel fan-out, `interrupt()` clarification, semantic cache, Alpha Vantage, LangSmith datasets.

UI is 25% of the grade, so step 9 is reserved real time and the Chat tab is usable early through the facade.

### 11.3 Testing strategy
Rules: the LLM and network are always mocked. No real API calls in tests. Coverage target is 100% of `src/` (CLAUDE.md).

| Layer | What tests cover |
|---|---|
| `core` / `utils` | config loading, `get_llm` role fallback and unknown provider error (patch the chat classes), usage tracker math |
| Market data | cache hit/miss, retry count and backoff (patched `sleep`), stale fallback, sample fallback labelling, freshness flags, currency present |
| RAG | chunk sizes and overlap, metadata survives to results, threshold gives "no match", manifest mismatch warning, tiny fixture index and a fake embedding |
| **(a) RAG evaluation harness** | the harness itself: hit-rate@k and MRR computed correctly on a tiny fixture; sweep runs over a parameter grid; off-topic questions score below the threshold on the fixture. Uses the fake embedding, no network |
| Agents | `BaseAgent` returns a plain answer and converts exceptions to error results; each agent with a fake LLM and stubbed tools; portfolio, market (return, volatility, drawdown, SMA, trend), goal and tax math as pure unit tests (tax values read from a test config); Tax Education retrieval is filtered to its topic |
| **(b) Nodes, state and reducers** | each node returns only its own result in a list; **after a two-agent run, both agents' results are present in `agent_outputs`, in order**; `errors` accumulates; single-writer fields are replaced; synthesizer reads the list and never fails on an empty `ok` list; **per-turn hygiene (same thread, checkpointer on):** a second turn never sees the first turn's results, `intent_hint` from a tab submit does not leak into the next chat message, `route_attempts` resets to 0 each turn, every result carries the current `turn_id`; `no_answer` only gives the synthesizer's no_answer message with the disclaimer, and all-`error` gives `fallback` |
| **(c) Router and routing** | **golden set of 30 to 40 labelled queries** (fake LLM returning the label under test, plus checks on the parsing, guardrail and `intent_hint` bypass); `specific_advice` always goes only to `safe_response`; re-routing works (`no_answer` with a ticker goes to `market`); **loop guard: never more than 2 hops**; all-failed goes to `fallback`; low confidence and `needs_input` go to `clarify` |
| **(d) Caching** | TTL expiry, stale fallback after failure, no caching past TTL (including market-dependent LLM answers), `CachedProvider` is transparent (same results and interface as the wrapped provider), exact-match LLM cache key includes role, model and prompt |
| Output guard (adversarial) | phrase check flags "you should sell", "I recommend buying", etc. and replaces the answer; a one-stock portfolio whose fake-LLM commentary says "you should sell" is caught through the Portfolio tab path (no router); clean educational text passes unchanged |
| Persistence | `ProfileRepository` round trip on a temp SQLite file; checkpointer keeps a thread across a new graph instance |
| Observability | LangSmith stays disabled by default; status line text; tags include thread id, agent and role (no network) |
| Web app | pure helpers (formatting, currency labels, validation, badge selection); one thin Streamlit smoke test using `streamlit.testing` |
| End to end | one fully mocked query through `run_query` per intent, including a clarify turn |

---

## 12. Persistence

Three stores, each with one job:

| Store | Holds | Lifetime |
|---|---|---|
| `st.session_state` | UI-only values: form inputs, selected tab, current `thread_id`, table edits | one browser session |
| Checkpointer (SQLite) | conversation state per `thread_id` | survives refresh and restart |
| Profile store (SQLite) | durable user facts | until the user clears them |

### 12.1 Short-term memory
- The LangGraph checkpointer moves from in-memory `MemorySaver` to SQLite (package `langgraph-checkpoint-sqlite`; **a NEW DEPENDENCY, see open questions**). Until approved, `MemorySaver` stays.
- The `thread_id` survives a browser refresh by storing it in the page URL query parameter (set through `st.query_params`) and reading it on load.

### 12.2 Long-term memory
- **`ProfileRepository` (Repository pattern)** on the same SQLite file. It stores risk appetite, goals, last portfolio and preferred currency. Agents call the repository and never touch SQL; tests swap in a temp file.
- **Example:** Goal Planning calls `repo.get_profile(name).risk_appetite` to pre-fill the risk profile; the Portfolio tab calls `repo.get_portfolio(name)` to reload saved holdings and `save_portfolio` on submit.

### 12.3 Identification and privacy
- A simple **profile name** entered in the sidebar. No authentication, so it is not suitable for sensitive data and the UI says so.
- Database path comes from config (`persistence.sqlite_path`). **`*.db` files are already in `.gitignore`** (nothing to add in a build step), so user data is never committed.

---

## 13. UI architecture

- **Layout:** `st.set_page_config(layout="wide")`; `st.tabs` with **Chat, Portfolio, Market, Goals**.
- **Responsive:** `st.columns` for side-by-side content (they stack on narrow screens); no fixed pixel widths; charts use container width.
- **Charts:** Plotly (already in requirements.txt): allocation pie, price history line, goal scenario lines.
- **Trust cues:** freshness badge on every price view and currency labels (USD or INR) on every price and value.
- **Sidebar:** profile name, session token usage (from `UsageTracker`), LangSmith status line.
- **Data flow:** the Chat tab calls the facade with no hint, so it goes through the router. Portfolio, Market and Goals tabs call the facade with an `intent_hint`, so they skip the router (section 4.2).
- Chat shows sources in an expander and the disclaimer once per answer; clarifying questions look like normal assistant turns.

**Mapping to rubric UI criteria**

| Criterion | Weight | Where it is met |
|---|---|---|
| Multi-tab and responsive | 10% | four `st.tabs`, wide layout, stacking columns, no fixed widths |
| Conversational flow | 8% | Chat tab with memory per thread, clarifying questions, sources expander, persistent thread across refresh |
| Visualisation | 7% | Plotly allocation, price history and goal scenario charts with freshness badge and currency labels |

---

## Decisions

- **Router model:** same as the default model (gpt-6-luna) for now; the `router` role exists so it can change by config only.
- **Market data layer:** lives in `src/data/market/`.
- **Market scope:** US and NSE (`.NS`); every value shows its currency.
- **Charts:** Plotly (already in requirements.txt).
- **Knowledge base topics:** stocks; bonds; ETFs and funds; risk and diversification; goals and retirement; accounts and taxes (India); market basics; glossary.
- **Agents:** all six are designed into the graph now; build status decides order.
- **Vector store:** FAISS for MVP, behind an interface.
- **Goal-planning and tax assumptions:** config values, labelled "illustrative" in the UI and answers, with an `as_of` date and a named source in config.
- **Clarification:** ordinary assistant reply for MVP; `interrupt()` is a stretch.
- **Acronym expansion (decisions the doc didn't specify):** a plain replacement table in config, not an LLM call; `merge_retrieval` searches every query as written and its expanded form (the expanded form replaces the acronym in place), but the expansion only FILLS: chunks found by a query as written come first with their own scores, a chunk found by both keeps the written query's score, and the expansion adds chunks only for the slots left up to `top_k` (diversity caps and the threshold apply to the combined list). Why: with the first version (best score per chunk, then sorted), "How does PPF work?" expanded to "public provident fund", which matched the title chunk `ppf-explained#0` (0.77, 39 characters) and the intro `#1`; those took both per-article slots and pushed out `#5` and `#21` (deposits and interest), so the Tax agent saw only a heading and an intro (the NO_ANSWER itself is inferred, no LLM was run). "What is LTCG?" and "What is STT?" are unchanged, because the written query finds little or nothing and the expansion fills the rest. `retrieval_check.py` prints scores before (`retrieve`) and after (`merge_retrieval`) with local embeddings only (no LLM, no API).
- **STT, LTCG and STCG in the Tax agent's topic:** short original-wording sections in `accounts_and_taxes_india/capital-gains-basics.md` (no rates, limits or dates; headings "STT and equity gains" and "Long-term and short-term capital gains (LTCG and STCG)"), because the glossary entries in `glossary/india-account-and-tax-terms.md` are in the `glossary` topic, which the Tax agent (locked to `accounts_and_taxes_india`) never searches; Finance Q&A and an all-topics search still find the glossary. The first heading was "Securities transaction tax (STT)": as a 35-character heading-only chunk it scored 0.32 on the near-miss question "How do I file my GST return?" (threshold 0.30) and raised the near-miss count from 4 to 5; the new heading scores 0.21. The two sections carry `<!-- VERIFY -->` markers (the KB now has 23 markers in 15 files). RAG eval (`run_rag_eval.py`, row 300/120/6/0.30, diversity on, index of 1332 chunks, old path = `select_chunks` alone): development hit rate 1.000 (0 misses of 121), MRR 0.865; held-out hit rate 1.000 (10 of 10), MRR 0.753; near-miss false positives 4; off-topic 0. Before the KB edit (1325 chunks) the same row was recorded as 1.000 / 0.865 / 1.000 and near-miss 4; its held-out MRR was not recorded.
- **Query expansion and the per-article cap (correction):** an earlier note here said the PPF and NPS swaps were harmless because the 2-chunks-per-article cap picked the higher-scoring pair. That was wrong: the picked pair was a title chunk and an intro, and "How does PPF work?" lost its answer. Fixed by the merge order above; tests in `tests/test_rag_retriever.py` cover the PPF case, an empty written result, a partial written result, a chunk found by both, and the threshold.
- **Eval on the production path and the merge sort key (Tax follow-ups 4):** `run_rag_eval.py` now measures `merge_retrieval()` over `retrieve()` (via `production_chunks()`), not `select_chunks()` alone, so acronym expansion is part of the numbers; `retrieve()` and `merge_retrieval()` take an optional `rag` settings dict (the eval passes its own top_k, threshold and caps; the agents pass none). On the new path the config row is unchanged: development 1.000 / MRR 0.865 (0 misses of 121), held-out 1.000 / MRR 0.753, near-miss 4, off-topic 0; the 400/120 and 400/60 rows are also unchanged. The merge sorts by (found by a query as written, score) through the optional `rank` argument of `select_chunks()`; the earlier `WRITTEN_FIRST` score bonus is gone. The eval prints the held-out and near-miss questions in full (`question_report`). The score-gap diagnostic searches the expanded form too, so the median answerable score reads 0.640 instead of 0.636. A test on the real saved index (`tests/test_rag_real_index.py`, skipped without the index or the cached model) guards the PPF, STT and LTCG cases.
- **Tax agent (decisions the doc didn't specify):** `small_savings_assumptions` is merged into `tax.exclusions.small_savings` (the `assumptions` list of a savings result is now empty); exclusions are per calculation and the tax-paid line only for equity asset types; short-term gains on property, unlisted and other assets are taxed at the slab rate the user enters (no config rate; `slab_rate_pct` is required, as for debt funds); long-term means strictly more than the configured months; the question path does not rewrite the query (one role, one LLM call) and ignores the Chat topic dropdown; all config figures are offered as facts (no relevance filter); an answer with no valid citation or an invented number is `no_answer` with `tax.discarded_message`, not returned uncited as Finance Q&A does; `[Fn]` citations are dropped when invalid but not renumbered; the footer with the label, as_of dates and tax-paid assumption is appended by code; the securities-transaction-tax assumption is listed first in the exclusions of every output; the agent name is `tax_education` (the router intent); the shared number guard (`numbers_in` / `collect_numbers`) reads Indian digit grouping and `N lakh` / `N crore` as N; `_validate_tax` raises `ConfigError` for any missing key the code reads and checks the asset-type lists are subsets of `asset_types`; the fact line shows the `display` text instead of value and unit; `merge_retrieval()` moved from `FinanceQAAgent` to `src/rag/retriever.py` so both agents share it; `is_number` is exported from `goal_inputs.py` for the tax input checks; tests use a synthetic `tax` config (`tests/tax_cfg.py`) so hand-computed cases survive when the real figures are corrected.
- **News agent (decisions the doc didn't specify):** citations are dropped when invalid but not renumbered, because the full numbered headline list is always shown; a summary with no valid citation counts as failed; the query wins over the ticker; a provider failure is `no_answer`, not `error`; only http(s) links are kept; news reuses the quote TTL (`market_data.cache_ttl_minutes`); `get_news` takes a search phrase; news cache keys and the fetched phrase are the same normalised text (casefold, collapsed whitespace), so the search sent to Google is lower-case; non-UTF-8 feeds are rejected rather than transcoded; the provider error classes live in `src/data/errors.py` (re-exported by `provider.py`) so the news layer does not import the market package.

- **Workflow skeleton (decisions the doc didn't specify, commit 1):** the graph is built by `build_graph(classify, agents=None, checkpointer=None)` in `src/workflow/graph.py` (Dependency Injection: tests pass fake agents and a fake `classify`; the router node itself is a thin wrapper, the real LLM router is the next commit); the pure routing functions are in `src/workflow/routing.py` and `edges.py`; `agents_to_run` is written by the router node and, for a re-route, by the `reroute` node (the doc said single writer: router); re-route candidates and the params they need come from `workflow.reroute` in config (`finance_qa` -> `market` needs `market_params`, `news` needs `news_params`), a candidate that already ran or is already queued is skipped; every agent node name equals its intent name and `workflow.hint_nodes` lists the accepted `intent_hint` values (an unknown hint is logged and ignored); `new_turn_input()` in `src/workflow/turn.py` builds the full fresh input and a test fails when a `GraphState` key is neither set there nor an append-reducer; all final nodes append the answer to `messages` as an `AIMessage`; `clarify` makes no LLM call (the agents' `needs_input` messages, or `workflow.messages.clarify_low_confidence`) and adds no disclaimer; `safe_response` uses static texts for now (an LLM-framed advice answer is the next-but-one commit); two or more ok results are joined VERBATIM under `workflow.agent_labels` headings with the `[n]` markers made unique (`merge_verbatim` in `src/workflow/answers.py`: shifted by the number of earlier sources, labelled for News, which has no source list), the LLM merge comes later and will fall back to this text; `wrong_agent` results are not presented (only `ok` and `no_answer` are); the MemorySaver import path was checked in `.venv` (`langgraph.checkpoint.memory.MemorySaver`, an alias of `InMemorySaver`) and structured output was checked with a mocked HTTP transport: `ChatOpenAI.with_structured_output(Schema)` (method `json_schema`) parses the reply, the usage callback still sees the token counts, and non-JSON output raises `pydantic.ValidationError`.
- **Router (decisions the doc didn't specify, commit 2):** `src/workflow/router.py` (`Router(llm=None)`, a callable `classify(state) -> Decision` for `build_graph`; the llm comes from `get_llm(role="router")`, tests inject a fake with `with_structured_output`). The prompt, aliases (Apple, Microsoft, Tesla, Reliance, TCS, Infosys, HDFC Bank, Nifty 50, Nifty), advice patterns and the history length are in `config.yaml` `workflow:`. An empty query is `clarify` without an LLM call. Too many agent intents is treated as an invalid reply (finance_qa) rather than truncated. A symbol typed in lower case that is not an alias is dropped. A low-confidence `out_of_scope` goes to `safe_response`, not `clarify`. `no_answer` results next to an `ok` result add a code-written note (`workflow.messages.no_answer_note`). Not verified: how well the real model classifies (`live_check_router.py` is written, not run).
- **Merge and safe_response (decisions the doc didn't specify, commit 3):** the two LLMs are created lazily (`LazyLLM` in `src/workflow/nodes.py`, role `synthesizer` / `safe_response`), so a graph that never merges or gives advice never calls the factory; `build_graph(..., synthesizer_llm, safe_response_llm, retrieve_fn)` accepts fakes. Output guard in the merge is a pattern check (a match means the verbatim join) rather than a reword call, because every agent answer was already guarded in `BaseAgent.run()`. A `[F2]`-style or `[News n]` marker that the model invents fails the merge, while an invalid plain `[n]` is dropped. `tests/conftest.py` now stops every test from reaching a real LLM or the real retriever through the workflow nodes (see STATUS).

## Open questions

1. Similarity-score threshold for the "no relevant source" path: tuned by the RAG evaluation harness (section 5.3).
2. Approval to add `langgraph-checkpoint-sqlite` (new dependency, required for SQLite short-term memory). `requirements.txt` is updated only after approval.
3. Verify each tax rate and limit in the `tax:` section of `config.yaml` against incometax.gov.in before the demo, and update its `as_of` date.
