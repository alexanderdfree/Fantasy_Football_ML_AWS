# Dashboard sources and client contracts

Edit `src/`; rebuild the committed serving artifact with `npm run build`.
`api.js` checks response version headers and the shared envelope before views
interpret forecast values. `api-contract.json` is exported from
`src.contracts.api`; `api-contract.js` carries the boundary's JSDoc types.
Legacy servers without headers remain readable, while unsupported major versions
fail visibly. Absent forecasts remain distinct from numeric zero.

Run the rendered dashboard checks locally:

```bash
npm ci
npm run build
npx playwright install chromium
npm test
```

Playwright serves the real dashboard bundle and vendored assets on localhost.
API fixtures are generated from local Flask routes; all external requests are
blocked. Tests cover scoring changes, degraded forecasts, missing/malformed
snapshots, player details, comparison metadata, and unsupported API versions.

From the repository root, refresh the shared contract and fixtures with:

```bash
python -m src.contracts.export
python -m ios.scripts.generate_client_fixtures
python -m ios.scripts.generate_comparison_fixture
```

Each command accepts `--check` for freshness verification. The Python fixture
tests verify real producer behavior; the narrow client CI workflow validates
the shared fixture envelopes, runs Chromium, and compiles/tests the Swift app.
