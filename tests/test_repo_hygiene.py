"""The agent brief must name the traps that do not raise."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_DOC = ROOT / "AGENTS.md"


def test_agents_md_names_the_traps_that_do_not_raise():
    assert AGENT_DOC.is_file(), "AGENTS.md is the single source of truth"
    text = AGENT_DOC.read_text(encoding="utf-8")
    for trap in [
        "FILTER_POLICY_VERSION",   # thresholds are hashed into the corpus id
        "native_sr",               # bandwidth is unreadable without it
        "provenance.json",         # where the policy version is recorded
        "requirements.txt",        # -e . cannot resolve oron-tts from an index
        "one thread",              # the pipeline is single-threaded by design
    ]:
        assert trap in text, f"AGENTS.md does not mention {trap!r}"


def test_claude_md_points_rather_than_copies():
    p = ROOT / "CLAUDE.md"
    assert p.is_file()
    text = p.read_text(encoding="utf-8")
    assert "AGENTS.md" in text
    assert len(text) < 1200, "it should delegate, not restate"


def test_requirements_txt_covers_every_declared_dependency():
    """It was empty once, so `pip install -r` was a silent no-op.

    Three separate pod runs died one missing import at a time before anyone
    noticed the file had nothing in it.
    """
    import re
    import tomllib

    req = {re.split(r"[><=\[]", line.strip())[0]
           for line in (ROOT / "requirements.txt").read_text().splitlines()
           if line.strip() and not line.startswith("#")}
    proj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {re.split(r"[><=\[]", d)[0]
                for d in proj["project"]["dependencies"]} - {"oron-tts"}
    assert not (declared - req), f"requirements.txt omits {sorted(declared - req)}"
