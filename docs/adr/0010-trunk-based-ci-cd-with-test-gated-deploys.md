# ADR-0010: Trunk-based CI/CD with test-gated deploys

**Status:** Accepted

**Decision.** All deployments happen from `main`. Three GitHub Actions workflows: [tests.yml](../../.github/workflows/tests.yml) runs on every push and PR; [batch-image.yml](../../.github/workflows/batch-image.yml) builds and registers a new Batch job definition revision when training code changes; [deploy.yml](../../.github/workflows/deploy.yml) builds and pushes the Flask image to ECS. Both deploy workflows gate on `tests.yml` passing.

**Context.** Personal project, single maintainer. Branching models designed for teams add ceremony without benefit. What's actually needed is a ratchet: broken code can't reach production, every push is traceable to a green test run, every image is tagged by SHA for rollback.

**Options considered.**

| Option | Ceremony | Rollback | Single-dev fit |
|---|---|---|---|
| Trunk-based + test-gated (chosen) | Low | SHA-tagged images | Excellent |
| Env branches (dev/staging/prod) | High | Revert + redeploy | Overkill |
| Manual deploy | None | Manual | Easy to skip tests |

**Chosen: trunk-based + test-gated.** Images are tagged by `${{ github.sha }}`; all historical tags stay in ECR for rollback. Batch job definitions are registered as new *revisions* (never deregistered), so rolling back is "submit a job with definition-name:revision-N-1."

**Rejected.** Environment branches would add a staging deploy with nothing behind it — for a personal project the "prod monitoring" is the dashboard on my laptop. Manual deploys were the original state; replacing them was the point.

**References.** [.github/workflows/tests.yml](../../.github/workflows/tests.yml), [batch-image.yml](../../.github/workflows/batch-image.yml), [deploy.yml](../../.github/workflows/deploy.yml). Landed in commit `ffb3119`.
