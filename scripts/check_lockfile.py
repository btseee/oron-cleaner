"""Verify requirements.lock still covers pyproject.toml.

Deliberately *not* a re-resolution. Recompiling and diffing would turn "some
upstream package released a version today" into a red build, which trains people
to ignore the signal. What actually needs catching is drift the repo controls: a
dependency added to pyproject.toml and never locked, or one locked at a version
the declared range excludes.

    python scripts/check_lockfile.py
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# oron-tts is a sibling repo rather than a published package, so it cannot
# appear in a resolved lockfile. It is pure stdlib and pulls nothing.
UNLOCKABLE = {"oron-tts"}

_NAME = re.compile(r"^([A-Za-z0-9._-]+)")
_SPEC = re.compile(r"([<>=!~]=?)\s*([0-9][^,;\s\]]*)")


def canonical(name: str) -> str:
    """PEP 503 normalisation: huggingface_hub and huggingface-hub are one name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(pyproject: Path) -> dict[str, list[tuple[str, str]]]:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    out: dict[str, list[tuple[str, str]]] = {}
    for raw in data["project"]["dependencies"]:
        dep = raw.split(";")[0].strip()          # drop environment markers
        name = canonical(_NAME.match(dep).group(1))
        out[name] = _SPEC.findall(dep)
    return out


def locked(lockfile: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in lockfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        out[canonical(_NAME.match(name).group(1))] = version.split()[0].strip()
    return out


def as_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in version.split("."):
        digits = re.match(r"\d+", chunk)
        if not digits:
            break
        parts.append(int(digits.group()))
    return tuple(parts) or (0,)


def satisfies(version: str, specs: list[tuple[str, str]]) -> bool:
    """Enough of PEP 440 for the ranges this project declares (>=, ==, <, <=)."""
    got = as_tuple(version)
    for op, bound in specs:
        want = as_tuple(bound)
        n = max(len(got), len(want))
        a = got + (0,) * (n - len(got))
        b = want + (0,) * (n - len(want))
        if op == ">=" and not a >= b:
            return False
        if op == ">" and not a > b:
            return False
        if op == "<=" and not a <= b:
            return False
        if op == "<" and not a < b:
            return False
        if op == "==" and a != b:
            return False
        if op == "!=" and a == b:
            return False
    return True


def main() -> int:
    lockfile = ROOT / "requirements.lock"
    if not lockfile.exists():
        print(f"{lockfile} is missing. See its header for how to generate it.")
        return 1

    want = declared(ROOT / "pyproject.toml")
    have = locked(lockfile)

    problems: list[str] = []
    for name, specs in sorted(want.items()):
        if name in {canonical(u) for u in UNLOCKABLE}:
            continue
        if name not in have:
            problems.append(f"  {name}: declared in pyproject.toml, absent from the lockfile")
            continue
        if not satisfies(have[name], specs):
            rendered = ",".join(f"{op}{v}" for op, v in specs)
            problems.append(f"  {name}: locked at {have[name]}, which is outside {rendered}")

    if problems:
        print("requirements.lock does not cover pyproject.toml:")
        print("\n".join(problems))
        print("\nRegenerate it -- the header of requirements.lock has the command.")
        return 1

    print(f"requirements.lock covers all {len(want) - len(UNLOCKABLE)} declared "
          f"dependencies ({len(have)} pinned in total).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
