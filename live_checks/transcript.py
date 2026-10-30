"""Saves what a live_check_*.py script prints to docs/live_checks/<script>-<YYYY-MM-DD>.txt.

Design pattern: DECORATOR (a wrapper that adds one behaviour and leaves the rest alone). `Tee` wraps sys.stdout: every
write still reaches the terminal unchanged, and a copy is kept. Example: `run_with_transcript(main, __file__)` in
live_check_workflow.py prints exactly what it printed before and also writes docs/live_checks/live_check_workflow-2026-10-30.txt.

The file is UTF-8, overwritten when the script runs again on the same day, and written even when the script fails
(a partial transcript is still evidence). Secrets are removed first: the values of environment variables whose name
contains KEY, TOKEN, SECRET or PASSWORD, `sk-...` style keys, `Bearer` tokens and any `Authorization:` header line.
Nothing is read from .env; the values come from the process environment only.
"""
from __future__ import annotations

import os
import re
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, TextIO

from src.utils.logger import get_logger

log = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPT_DIR = PROJECT_ROOT / "docs" / "live_checks"
REDACTED = "[redacted]"
SECRET_NAME = re.compile(r"KEY|TOKEN|SECRET|PASSWORD", re.IGNORECASE)
MIN_SECRET_LENGTH = 8                       # shorter env values ("1", "true") would blank out ordinary text
KEY_LIKE = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")
BEARER = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
AUTH_HEADER = re.compile(r"^.*\bauthorization\s*:.*$", re.IGNORECASE | re.MULTILINE)


def redact(text: str, environ: dict[str, str] | None = None) -> str:
    """The text with secrets replaced by [redacted]. Example: "key sk-abc..." -> "key [redacted]"."""
    env = os.environ if environ is None else environ
    for name, value in env.items():
        if SECRET_NAME.search(name) and len(value) >= MIN_SECRET_LENGTH:
            text = text.replace(value, REDACTED)
    text = AUTH_HEADER.sub(REDACTED, text)
    return BEARER.sub(REDACTED, KEY_LIKE.sub(REDACTED, text))


class Tee:
    """A text stream that passes every write to `target` unchanged and keeps a copy in `parts`."""

    def __init__(self, target: TextIO) -> None:
        self.target = target
        self.parts: list[str] = []

    def write(self, text: str) -> int:
        """Write to the real stream and remember the text."""
        self.parts.append(text)
        return self.target.write(text)

    def flush(self) -> None:
        """Flush the real stream."""
        self.target.flush()

    def __getattr__(self, name: str):
        """Everything else (encoding, isatty, ...) is the real stream's."""
        return getattr(self.target, name)


def transcript_path(script: str, today: date | None = None, out_dir: Path | None = None) -> Path:
    """docs/live_checks/<script name>-<date>.txt for a script path such as ".../live_check_qa.py"."""
    return (out_dir or TRANSCRIPT_DIR) / f"{Path(script).stem}-{(today or date.today()).isoformat()}.txt"


@contextmanager
def transcript(script: str, today: date | None = None, out_dir: Path | None = None) -> Iterator[Path]:
    """While active, stdout is copied; on exit the redacted copy is written to the transcript file (overwritten)."""
    path = transcript_path(script, today, out_dir)
    tee = Tee(sys.stdout)
    sys.stdout = tee
    try:
        yield path
    finally:
        sys.stdout = tee.target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(redact("".join(tee.parts)), encoding="utf-8", newline="\n")


def run_with_transcript(main: Callable[[], None], script: str) -> None:
    """Run `main()` with the transcript on. The printed output is unchanged; the file path goes to the log only."""
    with transcript(script) as path:
        log.info("Live check output will be saved to %s", path)
        main()
