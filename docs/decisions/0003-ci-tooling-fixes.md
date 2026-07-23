# 0003 — Reproducible type checking and a workable dependency audit

**Date:** 2026-07-23
**Status:** Accepted

## Context

First CI run on the new repo failed two ways.

**1. mypy passed on Python 3.11, failed on 3.12** — same source, same commit.

```
features.py:122: error: Argument 1 to "map" of "Index" has incompatible type
"Series[Any]"; expected "Mapping[Any, Hashable | None] | ..."
```

`pandas-stubs` was unpinned, so pip resolved a different stub version per
interpreter. The stricter stubs rejected `MultiIndex.map(Series)`.

**2. `pip-audit --strict` failed** with `gamestate: Dependency not found on
PyPI`. Our own package is installed editable (`pip install -e`) and has no PyPI
record. `--strict` treats "could not audit" as failure. There was no actual
vulnerability — a clean audit of the real dependencies returns
`No known vulnerabilities found`.

## Decisions

**Pin the stub packages.** `pandas-stubs` and `types-requests` are pinned
exactly in `[dev]`. A type check that varies by interpreter is not a check.
Version-drift belongs behind a deliberate bump, not a resolver coin-flip.

**Replace `MultiIndex.map` with explicit merges.** The stub error was pointing
at genuinely murky code. Two `merge(..., validate="one_to_one")` calls state
the join key plainly and assert cardinality — a bad join is one of the easiest
ways to produce a silently wrong model. Verified behaviour-preserving: log loss
0.6408, Brier 0.2250, sharpness 0.1405, skill 0.0885 before and after.

**`pip-audit --skip-editable`, without `--strict`.** The two flags conflict:
strict re-raises the skip as an error. pip-audit still exits nonzero on a real
vulnerability.

## Consequences

- Type checking is reproducible across the matrix.
- Stub bumps now require an explicit version change, which will surface as a
  reviewable diff rather than a mystery CI failure.
- Trade accepted: without `--strict`, an unauditable *third-party* dependency
  would pass quietly. Acceptable while every dependency is mainstream PyPI.
  Revisit by generating a lockfile and auditing that instead.

## What this validated

The 3.11/3.12 matrix caught a real reproducibility defect that a single-version
CI would have shipped. Keep the matrix.
