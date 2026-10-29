# AI Finance Assistant

Capstone project, graded by a rubric. Highest weights: multi-agent architecture,
LangGraph workflow, RAG, real-time data (40%), then UI (25%).

## Stack
Python, LangGraph/LangChain, FAISS, local sentence-transformers embeddings,
yfinance, Streamlit. LLM provider and model are set in config.yaml.

## Structure
src/agents, src/core, src/data, src/rag, src/web_app, src/utils, src/workflow, tests/

## Rules
- Read settings with get_config() and keys with get_secret(). Never hard-code values.
- Never read, print, or commit .env.
- Agents never create an LLM directly. They call get_llm() (Factory pattern).
- Every new module ships with unit tests in tests/. src coverage stays at 100%.
- Tests must mock the LLM and network calls. No real API calls in tests (limited budget).
- Use get_logger(), not print(). Add type hints and short docstrings.
- All user-facing financial answers carry the disclaimer from config.yaml.
  Educational content only, never specific investment advice.
- Ask before adding a new dependency. Keep requirements.txt updated.

## Source of truth
- docs/architecture.md is the spec. If the code must differ, change the doc in the same commit and list it under "Decisions the doc didn't specify".
- docs/STATUS.md is the handoff file. Read it at the start of every session. Update it at the end of every step.

## Evidence rules
- Any claim about behaviour must cite a file and function, or a command you ran. Anything you did not run or open is marked "not verified". Never state test counts, coverage or commit hashes from memory; run the command.
- Do not run live_check_*.py or anything that spends API money unless I ask.
- Never claim a library or API behaves a certain way without checking the installed version in .venv.

## Final report format (max 10 lines)
Files changed; tests (passed count, src coverage); commit hash; "Decisions the doc didn't specify"; open questions; anything marked "not verified". No recap of the steps taken.

## Commits
One commit per finished step with a clear message. Never leave the working tree dirty at the end of a step.

## Working style
- The user is learning. When you use a design pattern, name it in the docstring
  and explain it with a concrete example from this code.
- For tasks touching several files, propose a plan first.
- Prefer small, reviewable changes.

## Commands
- Run tests: pytest
- Check setup: python smoke_test.py