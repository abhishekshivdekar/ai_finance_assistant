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
- Every new module ships with unit tests in tests/. Target 80% coverage.
- Tests must mock the LLM and network calls. No real API calls in tests (limited budget).
- Use get_logger(), not print(). Add type hints and short docstrings.
- All user-facing financial answers carry the disclaimer from config.yaml.
  Educational content only, never specific investment advice.
- Ask before adding a new dependency. Keep requirements.txt updated.

## Working style
- The user is learning. When you use a design pattern, name it in the docstring
  and explain it with a concrete example from this code.
- For tasks touching several files, propose a plan first.
- Prefer small, reviewable changes.

## Commands
- Run tests: pytest
- Check setup: python smoke_test.py