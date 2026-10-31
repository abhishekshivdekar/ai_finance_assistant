"""Streamlit entry point. Run it with `streamlit run app.py` (or press F5 in VS Code: see .vscode/launch.json).

It sits at the repo root so that the repo root is on sys.path and `src` can be imported. All logic is in src/web_app.
"""
from src.web_app.main import render

render()
