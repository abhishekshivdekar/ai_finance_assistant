"""The live-check transcript helper (live_checks/transcript.py): copy of stdout, secrets removed, one file per day."""
import io
import sys
from datetime import date
from pathlib import Path

import pytest

from live_checks import transcript as tr

ROOT = Path(__file__).resolve().parent.parent
DAY = date(2026, 10, 30)


def test_the_path_is_the_script_name_and_the_date(tmp_path):
    assert tr.transcript_path("C:/x/live_check_qa.py", DAY, tmp_path) == tmp_path / "live_check_qa-2026-10-30.txt"
    assert tr.transcript_path("live_check_tax.py", None).name == f"live_check_tax-{date.today().isoformat()}.txt"
    assert tr.TRANSCRIPT_DIR == ROOT / "docs" / "live_checks"


def test_redact_removes_secret_values_keys_bearer_tokens_and_authorization_lines():
    env = {"OPENAI_API_KEY": "abc123456789", "HF_TOKEN": "tok-987654321", "SHORT_KEY": "1", "HOME": "abc123456789xyz", "DB_PASSWORD": "hunter2hunter2"}
    text = ("key abc123456789 and tok-987654321 and 1 stay\nkey sk-proj-ABCDEFGHIJKLMNOPQRSTUV end\n"
            "Authorization: Bearer secretvalue\nheader authorization : x\nBearer other.token done\nhunter2hunter2")
    out = tr.redact(text, env)
    for secret in ("abc123456789", "tok-987654321", "sk-proj-ABCDEFGHIJKLMNOPQRSTUV", "secretvalue", "other.token", "hunter2hunter2"):
        assert secret not in out
    assert out.count(tr.REDACTED) >= 6 and "and 1 stay" in out                    # a one-character secret value is not blanked out of the text


def test_redact_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv("DEMO_SECRET", "very-secret-value")
    assert tr.redact("is very-secret-value here") == f"is {tr.REDACTED} here"


def test_the_tee_passes_everything_through_and_keeps_a_copy():
    target = io.StringIO()
    tee = tr.Tee(target)
    assert tee.write("abc") == 3
    tee.flush()
    assert target.getvalue() == "abc" and tee.parts == ["abc"] and tee.getvalue() == "abc"       # other attributes come from the real stream


def test_a_run_prints_as_before_and_writes_a_utf8_file(tmp_path, capsys):
    with tr.transcript("live_check_demo.py", DAY, tmp_path) as path:
        print("hello ₹ world")
        print("second line")
    assert capsys.readouterr().out == "hello ₹ world\nsecond line\n"            # the terminal output is unchanged
    assert path.read_bytes().decode("utf-8") == "hello ₹ world\nsecond line\n" and sys.stdout is not None


def test_the_same_day_overwrites_and_a_failure_still_leaves_a_transcript(tmp_path, capsys):
    with tr.transcript("live_check_demo.py", DAY, tmp_path) as path:
        print("first run")
    with pytest.raises(RuntimeError):
        with tr.transcript("live_check_demo.py", DAY, tmp_path):
            print("second run")
            raise RuntimeError("boom")
    assert path.read_text(encoding="utf-8") == "second run\n" and list(tmp_path.iterdir()) == [path]
    assert "second run" in capsys.readouterr().out


def test_the_transcript_never_contains_a_secret(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-0123456789abcdef0123")
    with tr.transcript("live_check_demo.py", DAY, tmp_path) as path:
        print("calling with sk-test-0123456789abcdef0123")
    assert "sk-test" not in path.read_text(encoding="utf-8") and tr.REDACTED in path.read_text(encoding="utf-8")


def test_run_with_transcript_runs_main_and_saves_to_the_default_folder(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tr, "TRANSCRIPT_DIR", tmp_path)
    tr.run_with_transcript(lambda: print("done"), "live_check_demo.py")
    assert capsys.readouterr().out == "done\n"                                    # nothing extra is printed
    assert (tmp_path / f"live_check_demo-{date.today().isoformat()}.txt").read_text(encoding="utf-8") == "done\n"


@pytest.mark.parametrize("script", sorted(p.name for p in ROOT.glob("live_check_*.py")))
def test_every_live_check_script_saves_a_transcript(script):
    source = (ROOT / script).read_text(encoding="utf-8")
    assert "from live_checks.transcript import run_with_transcript" in source and "run_with_transcript(main, __file__)" in source
    assert "\n    main()" not in source.split('if __name__ == "__main__":')[-1]
