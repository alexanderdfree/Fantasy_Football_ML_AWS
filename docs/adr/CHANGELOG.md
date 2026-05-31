# ADR Changelog

Terse, chronological log of architecture changes — one line each: `YYYY-MM-DD · summary · (PR #N) · → ADR-00NN`. Full rationale lives in the per-decision
files in this directory; pre-split detail is in [../architecture-history.md](../architecture-history.md).

- 2026-05-31 · Split ARCHITECTURE.md into per-decision ADR files under docs/adr/; froze the prior Update history to architecture-history.md · → index in [../ARCHITECTURE.md](../ARCHITECTURE.md)
- 2026-05-31 · D12: launch-bound NN hot-path cuts (−15.6% QB) + `torch.compile` measured +169%/+62% (stays off on all archs) · (PR #655/#657) · → ADR-0012
- 2026-05-31 · D1: per-position `min_games_per_season` knob — RB/WR/TE→1, QB deferred; triggers a 6-position retrain · (PR #656) · → ADR-0001
