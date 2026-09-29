# 0006 — Pin the type checker with its stubs; drop Python 3.11

**Date:** 2026-09-28
**Status:** Accepted

## Context

Adding the database schema, the mypy pre-commit hook failed on four numpy
operations in previously-passing code (weighting.py, evaluate.py,
distribution.py), e.g.:

    weighting.py:50: error: Incompatible return value type
    (got "float", expected "ndarray[...]")   # 0.5 ** (age / half_life)

Local `mypy` in the 3.12 venv passed; the hook did not. Diagnosis took three
wrong turns before the cause was reproduced exactly, and the wrong turns are
worth recording so the mistake is not repeated:

1. **Wrong theory: Python version.** Guessed the hook analysed as 3.11, where
   numpy 2.5.1 stubs (PEP 695 `type` syntax) fail to parse. Added
   `language_version: python3.12`. No effect.
2. **Wrong theory: config inference.** Guessed `python_version` needed to be
   set in `[tool.mypy]`. Added it. No effect.
3. **Actual cause, reproduced.** Built a venv matching the hook's EXACT pins --
   `mypy==1.11.2` (from `rev: v1.11.2`) + `numpy==2.5.1` -- and the error
   reproduced immediately. mypy 1.11.2 (July 2024) is too old to understand
   numpy 2.5.1's stubs (2026): it mis-infers `scalar ** ndarray` as returning
   a scalar. mypy 1.18.2 against the same stubs: clean.

The lesson: the failing variable was the type checker's own version, not the
Python version and not the stubs alone. It was found only by reproducing
against the exact pinned versions instead of theorising -- which should have
been step one.

## Decisions

**Bump mypy to a version contemporary with the stubs.**
`.pre-commit-config.yaml` mypy hook: `rev: v1.18.2`. `pyproject.toml` [dev]:
`mypy>=1.18`. A type checker two years older than the stubs it reads is as
broken as an unpinned version -- age has to match, not just presence of a pin.

**Pin numpy alongside the other stub sources.** numpy ships its own stubs, so
its version IS a stub version. Pinned to 2.5.1 in [dev] (not core deps -- the
runtime numpy still floats). The pre-commit hook's `additional_dependencies`
mirrors all three: pandas-stubs, types-requests, numpy.

**Drop Python 3.11; standardise on 3.12.** `requires-python = ">=3.12"`, CI
matrix `["3.12"]`, hook `language_version: python3.12`. This is independent of
the bug above -- it is correct on its own merits: we develop, deploy, and run
on 3.12 exclusively. 3.11 was in the matrix only to demonstrate matrix testing
in week one, and it did earn its place then by catching the unpinned
pandas-stubs drift in ADR-0003. But numpy 2.5's stubs do require 3.12, and
carrying a version we never run is cost without benefit. With a single target,
setting `python_version = "3.12"` in `[tool.mypy]` is now correct -- the very
line ADR-0003 removed BECAUSE of the matrix. Dropping the matrix removed the
reason it was harmful.

## The general rule

Every environment that type-checks must agree on THREE things, not one:
  - the stub versions (pandas-stubs, types-requests, numpy);
  - the type checker version, contemporary with those stubs;
  - the Python version analysed.
Pinned in `pyproject.toml`, mirrored exactly in the pre-commit hook. This is
the third build break from toolchain drift (pandas-stubs and an over-eager
`python_version` in ADR-0003, mypy-vs-stubs here). The rule is now explicit so
there is not a fourth.

## Process note

Reproduce before theorising. All three failures in this session's diagnosis
came from proposing a fix before reproducing the error against the exact pinned
versions. A throwaway venv matching the lockfile would have found the cause in
one step instead of three. Adopt that as the default for any environment bug.

## Consequences

- One less CI job; faster pipeline.
- venv, pre-commit hook, and CI all type-check identically: mypy 1.18.2,
  numpy 2.5.1, Python 3.12.
- Reverting to 3.11 later would also require holding numpy below 2.5 for its
  stubs to parse -- not worth carrying pre-emptively.
