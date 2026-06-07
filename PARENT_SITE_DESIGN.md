# Parent Site Design — `alexfree.me` personal landing page

**Audience:** the agent (or human) who will build the **parent personal website** for `alexfree.me` in a *separate* repo.
**Status:** design / handoff spec. This document does **not** build the site — it specifies what to build and lists the concrete changes the **Fantasy Football ML** repo (this repo) must make so the two coexist.
**Owner:** Alexander Freeman (`alexanderdfree@gmail.com`).

---

## 1. Overview & goal

Today `alexfree.me` resolves straight to the **Fantasy Football ML** Flask app (DNS apex → ALB → ECS Fargate). The goal is to make `alexfree.me` a **personal landing site** (about / resume / projects / social links), and relocate the Fantasy project to its own subdomain.

**Decision: subdomain + redirect.**
- The parent personal site becomes the new root: `alexfree.me` (+ `www.alexfree.me`).
- The Fantasy app moves to **`fantasy.alexfree.me`**.
- `alexfree.me/fantasy` issues a **301 redirect** to `https://fantasy.alexfree.me` (so the URL the user mentioned still works and is discoverable).

This is the lowest-risk integration: the Fantasy app keeps owning a host root, so **no Fantasy application code changes** are needed (no Flask URL-prefix, no asset-path rewrites). Only DNS / TLS / infra plus a handful of string updates (Section 4).

### Target topology

```
                      ┌─────────────────────────────────────────────┐
   alexfree.me   ─────▶  CloudFront distribution                     │
   www.alexfree.me ───▶    • default behavior  ──▶ S3 (static parent site)
                      │     • viewer-request CF Function:            │
                      │         /fantasy*  ──▶ 301 → fantasy.alexfree.me
                      └─────────────────────────────────────────────┘

   fantasy.alexfree.me ─▶ ALB (HTTPS:443) ─▶ ECS Fargate (Fantasy Flask app, :8000)
                                              [UNCHANGED — already exists]
```

---

## 2. Architecture decision — why subdomain + redirect

The Fantasy serving app assumes it owns a host **root**:
- Templates and JS emit **root-relative** paths — `src/serving/templates/index.html` links `/static/css/style.css`, `/static/js/...`; `src/serving/static/js/app.js` `fetch()`es `/api/...`.
- Operational endpoints live at root: `/health` (the ALB health check), `/warm`, plus all `/api/*`.
- The ALB has **no path-based listener rules** — the default action forwards everything to the `fantasy-tg` target group.
- There is **no `APPLICATION_ROOT` / Blueprint prefix support** in the Flask app.

Therefore a *true* `alexfree.me/fantasy/*` subpath would require invasive, fragile changes (prefix-aware routing, rewriting every `/static` and `/api` reference in the template + `app.js`, and edge path rewriting). Rejected.

**Chosen:** keep Fantasy byte-identical at its own host (`fantasy.alexfree.me`) and let the **parent site's edge** own the `/fantasy` → subdomain redirect via a CloudFront Function (Section 3). Fantasy does not implement the redirect.

Rejected alternatives:
- **True subpath (`/fantasy/*` proxied to Fantasy):** invasive Fantasy changes (above).
- **CloudFront multi-origin with edge path-rewrite:** keeps Fantasy mostly unchanged but rewriting root-relative asset/`fetch` paths at the edge is brittle and hard to verify.

---

## 3. Parent site spec (what to build)

### 3.1 Stack
- **Static site** — plain HTML/CSS/JS, or a static generator (Astro / 11ty / Next static export) if preferred. **No server runtime.**
- Output is a folder of static assets uploaded to S3 and served via CloudFront.

### 3.2 Pages / sections
Single-page-with-anchors or multi-page are both fine. Required sections:
1. **Hero + About** — name, tagline, short bio, profile photo/avatar.
2. **Resume** — inline summary (education, experience, skills) and/or a link to a résumé PDF (host the PDF in the same S3 bucket).
3. **Projects** — **Fantasy Football ML featured first**, with:
   - Live demo link → `https://fantasy.alexfree.me`
   - Source link → `https://github.com/alexanderdfree/Fantasy_Football_ML_AWS`
   - Blurb (see Section 5).
4. **Social / Contact footer** — email + social icons/links (see Section 5).

### 3.3 Theme — reuse the Fantasy design tokens (visual continuity)
Pull these from `src/serving/static/css/style.css` so the parent site and the Fantasy dashboard feel like one product:

```css
:root {
  --bg-primary:   #0f1117;  /* page background (dark navy) */
  --bg-secondary: #1a1d27;
  --bg-card:      #21242f;  /* cards / panels */
  --bg-hover:     #282c3a;
  --border:       #2e3347;
  --text-primary:   #e8eaed;
  --text-secondary: #9aa0b0;
  --text-muted:     #6b7280;
  --accent:           #22c55e;  /* primary accent — green */
  --accent-secondary: #3b82f6;  /* secondary accent — blue */
  --radius:    8px;
  --radius-lg: 12px;
}
```
- **Font stack (system fonts, no web-font dependency):**
  `-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`
- **Header pattern:** bold accent-green wordmark + muted subtitle (mirrors Fantasy's `.logo` / `.subtitle` — `font-weight:700`, `color: var(--accent)`, `letter-spacing:-0.5px`).
- Dark mode only; cards use `--bg-card` on `--bg-primary` with `--border`.

### 3.4 Hosting / deploy
- **S3 bucket** for the static build. Prefer **private bucket + CloudFront Origin Access Control (OAC)** over the legacy S3 website-endpoint (OAC supports the redirect-via-CF-Function pattern and keeps the bucket private).
- **CloudFront distribution** in front of the bucket:
  - **Default cache behavior** → S3 origin (the static site).
  - **`/fantasy` redirect** → a **CloudFront Function** (viewer-request) attached to the default behavior (or a dedicated `/fantasy*` behavior):
    ```js
    function handler(event) {
      var uri = event.request.uri;
      if (uri === '/fantasy' || uri.startsWith('/fantasy/')) {
        return {
          statusCode: 301,
          statusDescription: 'Moved Permanently',
          headers: { location: { value: 'https://fantasy.alexfree.me' } }
        };
      }
      return event.request;
    }
    ```
  - **404 / SPA fallback:** map 403/404 to `/index.html` (custom error responses) if using a SPA router; for a plain multi-page static site, configure a real `404.html`.
- **GitHub Action** to deploy: build → `aws s3 sync ./build s3://<bucket> --delete` → `aws cloudfront create-invalidation --distribution-id <id> --paths "/*"`.

### 3.5 DNS (Namecheap) — the cutover
The domain is registered at **Namecheap** (authoritative DNS; not Route 53). In *Advanced DNS*:
- Re-point apex `ALIAS @` and `CNAME www` **from the ALB to the CloudFront** distribution domain.
- This is the **cutover step** that moves the root away from Fantasy — do it **after** `fantasy.alexfree.me` is confirmed live (Section 4 sequencing).

### 3.6 TLS
- New **ACM certificate in `us-east-1`** (CloudFront requires `us-east-1` regardless of bucket region) covering **`alexfree.me` + `www.alexfree.me`**, **DNS-validated** (add the ACM CNAME records in Namecheap). Attach to the CloudFront distribution as an alternate domain name (CNAME).

---

## 4. Fantasy-side integration hooks (checklist for THIS repo)

These are the concrete changes the Fantasy repo must action. **No application code changes** — infra + strings only.

| # | Change | Where |
|---|--------|-------|
| 1 | Add DNS record `CNAME fantasy → <ALB DNS name>` | Namecheap Advanced DNS |
| 2 | Add `fantasy.alexfree.me` as a SAN on the ACM cert (or issue a new cert) and attach to the ALB **HTTPS:443 listener** | ACM + ALB. Cert requested in `infra/aws/bootstrap.sh` Step 7; 443 listener created Step 8 |
| 3 | `DOMAIN="alexfree.me"` → `DOMAIN="fantasy.alexfree.me"`; drop the `DOMAIN_WWW="www.alexfree.me"` line (www belongs to the parent now) | `infra/aws/bootstrap.sh` line 34–35; mirror in `infra/aws/teardown.sh` + `infra/aws/README.md` |
| 4 | `SERVICE_URL: https://alexfree.me` → `https://fantasy.alexfree.me` (post-deploy/post-train `/warm` probe target) | `.github/workflows/deploy.yml` (~line 52) **and** `.github/workflows/train-batch.yml` (~line 68) |
| 5 | Dashboard URL `alexfree.me` → `fantasy.alexfree.me` | `README.md` lines 3, 172, 206 |

**No-touch (verified):** `/health` (ALB health check path), `/warm`, `/static/*`, `/api/*` all keep working unchanged at the new host. The `alexfree.me` mentions in `src/serving/app.py` (lines 815, 3106) are **dated incident comments** — leave them.

**Sequencing (avoid an outage window):**
1. Stand up `fantasy.alexfree.me` first: add the cert SAN (#2), attach to the 443 listener, add the `fantasy` CNAME (#1).
2. Confirm `https://fantasy.alexfree.me/health` returns 200 and the dashboard loads.
3. **Only then** re-point apex/`www` to CloudFront (Section 3.5) and add the `/fantasy` redirect.

This ordering keeps Fantasy reachable at its new host throughout the cutover, so the apex move never strands it.

---

## 5. Personal content (known facts + placeholders)

### Pre-filled (known)
- **Name:** Alexander Freeman
- **Email:** `alexanderdfree@gmail.com`
- **GitHub:** `https://github.com/alexanderdfree`
- **Featured project — Fantasy Football ML**
  - Live: `https://fantasy.alexfree.me`
  - Source: `https://github.com/alexanderdfree/Fantasy_Football_ML_AWS`
  - Blurb (from `README.md`): *"A per-position machine learning system that predicts weekly NFL fantasy points for QBs, RBs, WRs, TEs, Kickers, and D/STs — comparing a Ridge baseline, a custom PyTorch multi-head neural network (with an attention variant at every position), and LightGBM across the 2012–2025 seasons. Ships GPU training on AWS Batch, ECS Fargate serving, and CI/CD that gates every push."*

### Placeholders — fill before launch (`<!-- TODO -->`)
- `<!-- TODO: profile photo / avatar -->`
- `<!-- TODO: tagline + bio prose -->`
- `<!-- TODO: resume — education, experience, skills (or résumé PDF link) -->`
- `<!-- TODO: LinkedIn URL -->`
- `<!-- TODO: X/Twitter URL -->`
- `<!-- TODO: any other socials (Bluesky, Mastodon, personal blog, etc.) -->`

---

## 6. Verification (run after cutover)

```bash
curl -sI https://alexfree.me/            # 200, server: CloudFront — parent site
curl -sI https://alexfree.me/fantasy     # 301, Location: https://fantasy.alexfree.me
curl -sI https://fantasy.alexfree.me/health   # 200 — Fantasy app healthy at new host
curl -sI https://www.alexfree.me/        # 200 (or 301→apex) — parent site
```
Then browser-load `https://alexfree.me`: the dark-themed landing page renders, the Projects → Fantasy link opens the dashboard, and the social/contact links resolve.

---

## 7. Reference — Fantasy facts this doc relies on

| Fact | Source |
|------|--------|
| Header/nav/branding ("Fantasy Football Predictor", `.logo`/`.subtitle`) | `src/serving/templates/index.html` |
| Design tokens (colors/fonts/radius) reused in Section 3.3 | `src/serving/static/css/style.css` |
| DOMAIN vars, ACM (Step 7), HTTPS listener (Step 8) | `infra/aws/bootstrap.sh`, `infra/aws/teardown.sh`, `infra/aws/README.md` |
| `SERVICE_URL` warm-probe target | `.github/workflows/deploy.yml`, `.github/workflows/train-batch.yml` |
| Dashboard URL references | `README.md` (lines 3, 172, 206) |
| ALB health check path `/health`; container port 8000; Fargate (arm64) → ALB + ACM HTTPS | `infra/aws/` + `gunicorn.conf.py` + `README.md` line 206 |
