# ADR-0023 — React frontend built with esbuild into a committed bundle

**Status:** Accepted (2026-07-05)

## Decision

Rewrite the serving dashboard frontend (formerly a single 2,290-line vanilla-JS
`app.js`) as React function components under `src/serving/frontend/`, adopting
the Fantasy Football Predictor design system (tokens, OLED night mode, and its
React component primitives under `frontend/src/ds/`). The bundle is built by
esbuild (`npm run build` in `src/serving/frontend/`) and **committed at the old
path `src/serving/static/js/app.js`** — the serving runtime, Dockerfile, and
deploy pipeline remain Node-free.

## Context

The uploaded design system shipped a full React recreation of the dashboard
(new OLED theme, richer filtering, a Changelog & Timeline tab) and the owner
chose a full React rewrite over a CSS-only retrofit. Constraints discovered up
front:

- `tests/test_app.py::TestStaticAssets` forbids external `<script>` sources on
  `/` (Chart.js is vendored for exactly this reason) — CDN React is out.
- The serving image is single-stage `python:3.12-slim` + `uv` (ADR-0008/D10);
  the pre-PR hook and CI pytest shards run without Node.
- `tests/test_app.py` greps the served `/static/js/app.js` for the literal
  ESPN-combiner resize string — the bundle must keep that path and survive
  minification with the literal intact.
- `.gitignore` ignores `dist/`/`build/` globally, so the build output cannot
  live in a directory with those names.

## Chosen

- **esbuild** (`build.mjs`): `--bundle --jsx=automatic --minify-whitespace
  --minify-syntax` with **identifier minification off** so the pinned
  headshot-combiner literal survives byte-for-byte; React 18.3 from npm is
  bundled in; Chart.js stays the vendored `window.Chart` global (never
  bundled), keeping `test_vendored_chartjs_is_served` authoritative.
- **Committed bundle** at `src/serving/static/js/app.js` (same idiom as the
  vendored Chart.js and the committed `comparison_experts.json`): pytest, the
  pre-PR hook, the Dockerfile `COPY src/ src/`, and local `python -m
  src.serving.app` all keep working with no Node toolchain.
- **CI staleness guard**: the `frontend-bundle` job in `tests.yml` rebuilds the
  bundle whenever `src/serving/frontend/**` or the bundle changed and fails on
  any diff — a `.jsx` edit cannot ship without its rebuilt artifact.
  `tests-pass` requires it.
- **Sources under `src/serving/frontend/`** so `scope_positions.py` keeps
  scoping frontend changes to the `serving` test shard (a root `package.json`
  would fan out all 8 shards) and `deploy.yml`'s `src/**` filter redeploys on
  frontend changes. `node_modules/` and the source dir are `.dockerignore`d.

## Rejected

- **Multi-stage Docker Node builder** (bundle not committed): breaks the local
  pre-PR `pytest -m unit` and the CI serving shard (both serve `/static/js/
  app.js` in tests), puts Node in the image build critical path, and violates
  the two-image philosophy (ADR-0008).
- **Vendored React UMD + Babel-standalone / hand-transpiled views** (the design
  system kit's own loading mode): prototyping-grade — per-page-load transpile
  cost, no tree-shaking, unmaintainable at ~2,500 lines of views.
- **CSS-tokens-only retrofit keeping vanilla JS**: rejected by the owner in
  favor of the full rewrite (the design system's React primitives power the
  upcoming filter-bar/timeline features).

## Consequences

- Frontend edits happen in `src/serving/frontend/src/**` and require
  `npm run build` before commit (SETUP.md § Frontend build); the CI guard
  catches a stale bundle.
- The React views preserve the production DOM structure and class names, so
  `style.css` (design-system tokens, ADR-0023 companion reskin) styles both
  the old and new runtime identically and no test pins changed.
- JS has no lint/format gate (ruff is Python-only); the frontend dir is
  excluded from ruff and formatted by convention (4-space, double quotes).

## References

- `src/serving/frontend/build.mjs` (build contract), `.github/workflows/tests.yml`
  (`frontend-bundle` job), `tests/test_app.py::TestStaticAssets` (pinned assets).

## Changelog

- 2026-07-05 · Initial decision: React + esbuild committed-bundle frontend,
  design-system adoption (midnight/OLED themes), CI staleness guard.
