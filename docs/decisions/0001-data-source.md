# 0001 — Use nflverse as the initial data source

**Date:** 2026-07-22
**Status:** Accepted

## Context

The engine needs game-level and eventually play-level NFL data going back far
enough to backtest. Two viable paths: consume a maintained open dataset, or
scrape primary sources (Pro Football Reference, ESPN, the NFL's own feeds).

## Options considered

**nflverse.** Community-maintained, cleaned, published as CSV/parquet on GitHub
releases. Game table covers 1999–present and already includes scores, closing
betting lines, weather, roof/surface, rest days, and coaching. Play-by-play
available separately with EPA and CPOE precomputed.
*Cost:* we inherit someone else's cleaning decisions without seeing them, and
we depend on a volunteer project's continued maintenance.

**Scraping.** Full control over what we collect and how it is parsed. Teaches
the realities of data acquisition — HTML drift, rate limits, retries, politeness.
*Cost:* realistically a month before the first model runs, and every hour of it
is spent on parsing rather than on the problems this project is actually about.

## Decision

nflverse, with the ingestion layer isolated behind `ingest.fetch_games()` so an
alternative source can be substituted without touching anything downstream.

## Consequences

- A working backtest exists in week one instead of week five.
- We are exposed to upstream schema changes; the raw parquet cache limits the
  blast radius, since a bad upstream push cannot silently overwrite good data.
- Scraping remains a deliberate later exercise, not a prerequisite.
- Betting lines arrive free, which matters: the market baseline is the honest
  benchmark for this whole project, and building it was a one-line addition.

## Revisit if

We need data nflverse does not publish — snap counts by alignment, route
participation, or anything requiring a paid provider (PFF, Next Gen Stats).
