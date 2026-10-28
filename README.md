# AI Finance Assistant

Multi-agent financial education assistant (LangGraph + RAG + Streamlit).

> Work in progress. Full README (architecture, setup, API docs, troubleshooting) is written on Day 13.

## Quick start
1. Create a virtual environment and install `requirements.txt`
2. Copy `.env.example` to `.env` and add your `GOOGLE_API_KEY`
3. Run `smoke_test.py`

## RAG tuning results

The retrieval settings were chosen with the evaluation harness in `tests/rag_eval/` (run with `run_rag_eval.py`; it uses the local embedder, so it costs no API tokens). The harness uses the same `select_chunks()` function as the app, so it measures what the app does. Scores are the share of questions whose labelled article is in the top `top_k` (hit rate) and the mean reciprocal rank of the first correct article (MRR). The development set has 121 questions; the held-out set has 10 independent questions.

| chunk_size | chunk_overlap | top_k | threshold | dev hit rate | dev MRR | held-out hit rate | held-out MRR |
|---|---|---|---|---|---|---|---|
| **300** | **120** | **6** | **0.30** | **1.000** | **0.865** | **1.000** | **0.753** |
| 400 | 120 | 6 | 0.30 | 0.992 | 0.851 | 0.900 | 0.750 |
| 400 | 60 | 6 | 0.35 | 0.992 | 0.858 | 0.900 | 0.700 |

Each row was scored with the diversity caps on and off; the numbers were identical, so the table shows one value. No off-topic question matched anything in any row.

**Why 300 / 120 / 6 / 0.30 won:** it has the best hit rate and the best MRR on both the development set and the held-out set, and it lets through no more false matches than the others. A likely reason is that smaller chunks keep each chunk closer to one idea, but that was not tested separately. The settings are applied in `config.yaml` (glossary keeps its own 250/0 chunking).

**Held-out caveat:** ten questions is a small sample. One question is worth 0.10 of hit rate, so the held-out gap between 300/120 and the other two rows is a single question ("What happens to my money if the stock market crashes tomorrow?"). The held-out set also informed the final choice, so treat its scores as indicative, not a guarantee.

**Near-miss limitation:** the threshold does not separate finance-adjacent questions that the knowledge base does not cover (for example loans, crypto or insurance) from answerable ones: 4 of 5 such near-miss questions still retrieve something above 0.30 in every row. Raising the threshold enough to stop this would also drop correct answers, so retrieval alone cannot solve it. The defence is layered: (1) a grounded prompt that tells the model to answer only from the retrieved context and to say so when the context does not cover the question, and (2) the router's `out_of_scope` intent, which sends questions outside the assistant's topics to a safe response before retrieval is used (architecture sections 3.3 and 4). These two layers belong to the agent and router steps and are not built yet.

*Educational purposes only. Not financial advice.*
