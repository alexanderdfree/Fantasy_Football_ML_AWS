> Historical record. Validate current code, configuration and ADRs before applying the recorded fix.

### [FIXED] Gunicorn `--preload` pre-warm broke ALB health checks during task replacement
- **Files:** gunicorn launch config / Dockerfile CMD. Tried in PR #148 (`d69f427`), reverted in PR #149 (`8ff26be`).
- **What:** To prevent 503s on ECS task replacement, PR #148 tried pre-warming the data + model caches at module import time under `gunicorn --preload`. Under `--preload`, the import happens *before* the worker binds its socket — so the ALB's TCP health check saw connection-refused for the entire pre-warm duration and marked the new task unhealthy before it could serve a single request. The intended fix (skip the 503 window) created an "unhealthy task" window that was strictly worse.
- **Fix:** Reverted (`8ff26be`). The right place for pre-warm work is a `post_fork` hook or a background thread that fires *after* the worker binds its socket — that path hasn't been re-attempted yet.
- **Lesson:** Under `gunicorn --preload`, module-import work runs before `bind()`. Anything slower than the ALB's TCP health-check timeout will produce a TCP-refused window that fails the deploy. Pre-warm in `post_fork` or a background thread, not at module import. (Captured in auto-memory as the "no module-level pre-warm under --preload" rule.)
