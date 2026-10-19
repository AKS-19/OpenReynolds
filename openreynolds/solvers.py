"""Which solver a study runs: OpenFOAM on a CPU workspace, or Felix on cloud GPUs.

A study's solver is fixed once and holds for the whole study, resumes included. It is
either given (`--solver openfoam|felix`) or left open (`--solver auto`, the default),
and then chosen from the person's first message by a request of its own, before the
main agent's first (`desk.route`); no workspace is started until it has been, because a
workspace has one kind and a study binds only to a workspace of its own kind. The main
agent never makes the choice: its prompt and tools are its solver's from the first
request, and a session's prompt and tools do not change under it.

A study recorded before solvers existed has no solver on its record and is an
OpenFOAM study. One that was left open and never routed -- nobody said anything -- is
recorded as `auto`, and is routed on its first message when it is resumed.
"""

from __future__ import annotations

OPENFOAM = "openfoam"
FELIX = "felix"
AUTO = "auto"

SOLVERS = (OPENFOAM, FELIX)
CHOICES = (AUTO, OPENFOAM, FELIX)


def normalize(value: str | None) -> str | None:
    """`auto`, `openfoam` or `felix` from any spelling of one, else None."""
    text = (value or "").strip().lower()
    return text if text in CHOICES else None


def chosen(value: str | None) -> bool:
    """Whether `value` names a solver, rather than the choice still to be made."""
    return normalize(value) in SOLVERS
