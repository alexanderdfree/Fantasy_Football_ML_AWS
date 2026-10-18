import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

const fixtures = JSON.parse(readFileSync(new URL("../../../../ios/Tests/Fixtures/client_contract.json", import.meta.url)));
const comparison = JSON.parse(readFileSync(new URL("../../../../ios/Tests/Fixtures/comparison_current.json", import.meta.url)));

async function localAPI(page, { snapshotStatus = 200, snapshot = fixtures.snapshot, playerStatus = 200, version = "1.0" } = {}) {
    // No production/CDN traffic: the actual app assets and all fixtures are local.
    await page.route("**/*", async (route) => {
        const url = new URL(route.request().url());
        if (url.origin !== "http://127.0.0.1:4173") return route.abort();
        if (!url.pathname.startsWith("/api/")) return route.continue();
        let status = 200;
        let body;
        const scoring = url.searchParams.get("scoring") || "ppr";
        if (url.pathname === "/api/snapshot") {
            status = snapshotStatus;
            body = status === 200 ? snapshot : { error: "snapshot not available" };
        } else if (url.pathname === "/api/predictions") body = fixtures.predictions[scoring];
        else if (url.pathname === "/api/weeks") body = { weeks: [1] };
        else if (url.pathname === "/api/teams") body = { teams: ["KC"] };
        else if (url.pathname === "/api/comparison") body = comparison;
        else if (url.pathname === "/api/model_architecture") body = fixtures.architecture;
        else if (url.pathname.startsWith("/api/player/")) {
            status = playerStatus;
            body = status === 200 ? fixtures.player[scoring] : { error: "Player unavailable" };
        } else if (url.pathname === "/api/upcoming_week") body = { available: false, reason: "offseason" };
        else body = {};
        return route.fulfill({ status, contentType: "application/json", headers: { "X-FFP-Contract-Version": version }, body: JSON.stringify(body) });
    });
}

function dstSnapshot(teams) {
    const snapshot = structuredClone(fixtures.snapshot);
    for (const scoring of Object.keys(snapshot.scoring)) {
        const base = snapshot.scoring[scoring].find((player) => player.position === "DST");
        snapshot.scoring[scoring] = teams.map((team) => ({
            ...base, player_id: team, name: `${team} D/ST`, team,
            // A stale or unavailable API headshot must not break the local logo.
            headshot: `https://a.espncdn.com/unavailable/${team}.png`,
        }));
    }
    return snapshot;
}

test("all team logos and historical aliases decode with external traffic blocked", async ({ page }) => {
    const teams = "ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS JAC LAR OAK SD STL WSH".split(" ");
    await localAPI(page, { snapshot: dstSnapshot(teams) });
    await page.goto("/#predictions");
    const logos = page.locator("#predictions-body .team-logo");
    await expect(logos).toHaveCount(teams.length);
    for (const logo of await logos.all()) {
        await logo.scrollIntoViewIfNeeded();
        await expect(logo).toBeVisible();
        await expect.poll(() => logo.evaluate((image) => image.complete && image.naturalWidth === 200)).toBe(true);
    }
});

for (const [team, playerStatus] of [["NO", 200], ["NYG", 404]]) {
    test(`${team} DST detail uses a local logo with player response ${playerStatus}`, async ({ page }) => {
        const snapshot = dstSnapshot([team]);
        for (const rows of Object.values(snapshot.scoring)) {
            rows[0].opponent = team === "NO" ? "NYG" : "NO";
            rows[0].is_home = 1;
        }
        await localAPI(page, { snapshot, playerStatus });
        await page.route("**/api/upcoming_week", (route) => route.fulfill({
            contentType: "application/json",
            body: JSON.stringify({ available: true, season: 2026, week: 6,
                generated_at: new Date().toISOString(), scoring: snapshot.scoring }),
        }));
        if (playerStatus === 200) {
            await page.route("**/api/player/**", (route) => route.fulfill({
                contentType: "application/json",
                body: JSON.stringify({ ...fixtures.player.ppr, position: "DST", team, name: `${team} D/ST`, headshot: "" }),
            }));
        }
        await page.goto("/");
        const row = page.locator(`#homepage-body [data-player-id="${team}"]`);
        await expect(row.locator(".team-logo")).toHaveCount(2);
        await expect.poll(() => row.locator(".team-logo").evaluateAll((images) => images.every((image) => image.complete && image.naturalWidth === 200))).toBe(true);
        await row.locator(".col-player").click();
        await expect(page.locator("#modal-name")).toHaveText(`${team} D/ST`);
        const logo = page.locator("#modal-headshot");
        await expect(logo).toBeVisible();
        await expect.poll(() => logo.evaluate((image) => image.complete && image.naturalWidth === 200)).toBe(true);
        if (playerStatus === 404) await expect(page.locator("#modal-note")).toContainText("No prior-season");
    });
}

test("a failed logo does not hide a replacement team's image", async ({ page }) => {
    await localAPI(page, { snapshotStatus: 404 });
    let team = "NO";
    await page.route("**/api/predictions?**", (route) => route.fulfill({
        contentType: "application/json",
        body: JSON.stringify({ ...fixtures.predictions.ppr,
            players: [{ ...dstSnapshot([team]).scoring.ppr[0], player_id: "same-row" }], total: 1 }),
    }));
    await page.route("**/static/img/teams/no.png", (route) => route.abort());
    await page.goto("/#predictions");
    const logo = page.locator('[data-player-id="same-row"] .team-logo');
    await expect(logo).toHaveAttribute("src", "/static/img/teams/no.png");
    await expect(logo).toBeHidden();
    team = "NYG";
    await page.locator("#scoring-filter").getByRole("button", { name: "Standard", exact: true }).click();
    await expect(logo).toHaveAttribute("src", "/static/img/teams/nyg.png");
    await expect(logo).toBeVisible();
    await expect.poll(() => logo.evaluate((image) => image.complete && image.naturalWidth === 200)).toBe(true);
});

test("history replaces a failed presentation when accepted results arrive", async ({ page }) => {
    await localAPI(page);
    let accepted = false;
    await page.route("**/api/benchmark_history", route => {
        const value = accepted ? 1 : 9;
        return route.fulfill({
            contentType: "application/json",
            headers: { "X-FFP-Contract-Version": "1.0" },
            body: JSON.stringify({ repo_slug: "owner/repo", rows: [{
                run_id: accepted ? "accepted-row" : "failed-row", training_run_id: "same-run",
                timestamp: "2026-09-10T01:00:00", git_hash: "abcdef0", positions: ["QB"],
                validation_status: accepted ? "accepted" : "validation_failed",
                ridge: [{ position: "QB", mae: value }], nn: [], attn_nn: [], lgbm: [],
                total_elapsed_sec: 1,
            }] }),
        });
    });
    await page.goto("/#history");
    await expect(page.locator("#view-history")).toContainText("Validation failed");
    await expect(page.locator("#history-table-container tbody tr")).toHaveCount(1);
    accepted = true;
    await page.evaluate(() => window.dispatchEvent(new Event("focus")));
    await expect(page.locator("#view-history")).not.toContainText("Validation failed");
    await expect(page.locator("#history-table-container tbody tr")).toHaveCount(1);
});

test("snapshot scoring switches and degraded forecasts preserve null versus zero", async ({ page }) => {
    await localAPI(page);
    await page.goto("/#predictions");
    const qb = page.locator('[data-player-id="fixture-QB"]');
    await expect(qb.locator(".col-actual")).toHaveText("30.0");
    await page.locator("#scoring-filter").getByRole("button", { name: "Standard", exact: true }).click();
    await expect(qb.locator(".col-actual")).toHaveText("10.0");
    await expect(page.locator("#degraded-banner")).toContainText("TE");
    await expect(page.locator('[data-player-id="fixture-TE"] .col-pred.ridge-col')).toHaveText("--");
    await expect(page.locator('[data-player-id="fixture-K"] .col-pred.ridge-col')).toHaveText("0.0");
});

test("missing snapshot falls back to live predictions and scoring stays aligned", async ({ page }) => {
    await localAPI(page, { snapshotStatus: 404 });
    await page.goto("/#predictions");
    await expect(page.locator('[data-player-id="fixture-QB"] .col-actual')).toHaveText("30.0");
    await page.locator("#scoring-filter").getByRole("button", { name: "Half PPR", exact: true }).click();
    await expect(page.locator('[data-player-id="fixture-QB"] .col-actual')).toHaveText("20.0");
});

test("malformed snapshot cannot silently show an empty scoring format", async ({ page }) => {
    const snapshot = structuredClone(fixtures.snapshot);
    delete snapshot.scoring.standard;
    await localAPI(page, { snapshot });
    await page.goto("/#predictions");
    await page.locator("#scoring-filter").getByRole("button", { name: "Standard", exact: true }).click();
    await expect(page.locator('[data-player-id="fixture-QB"] .col-actual')).toHaveText("10.0");
});

test("player detail follows the selected scoring and renders the actual chart", async ({ page }) => {
    await localAPI(page);
    await page.goto("/#predictions");
    await page.locator("#scoring-filter").getByRole("button", { name: "Standard", exact: true }).click();
    await page.locator('[data-player-id="fixture-QB"] .col-actual').click();
    await expect(page.locator("#modal-name")).toHaveText("Fixture QB");
    await expect(page.locator("#modal-avg")).toHaveText("10.0");
    await expect(page.locator("#player-chart")).toBeVisible();
});

test("server failures in player detail remain errors", async ({ page }) => {
    await localAPI(page, { playerStatus: 500 });
    await page.goto("/#predictions");
    await page.locator('[data-player-id="fixture-QB"] .col-actual').click();
    await expect(page.locator("#modal-name")).toHaveText("Error loading player");
    await expect(page.locator("#modal-note")).not.toContainText("No prior-season");
});

test("comparison displays shared components, cohorts, ESPN and partial reference", async ({ page }) => {
    await localAPI(page);
    await page.goto("/#comparison");
    await expect(page.locator("#comparison-contract")).toContainText("same regular-season player-weeks");
    await expect(page.locator("#comparison-contract")).toContainText("NFL.com does not supply matching field-goal yardage");
    await expect(page.locator("#view-comparison")).toContainText(comparison.cohort_definitions.weekly_reference_top24);
    await expect(page.locator("#comparison-weekly-top24")).toContainText("partial reference");
    const allTable = page.locator("table").filter({ has: page.locator("#comparison-all-body") });
    await expect(allTable.getByRole("columnheader", { name: "ESPN" })).toBeVisible();
    // NFL.com offense is excluded server-side, so its all-dash column is not rendered.
    await expect(allTable.getByRole("columnheader", { name: "NFL.com" })).toHaveCount(0);
    await expect(page.locator("#comparison-contract")).toContainText("reproduces RotoWire");
    // Headline cohort and the paired-interval verdict per row.
    await expect(page.locator("#comparison-depth-starters tr")).toHaveCount(6);
    await expect(page.locator("#comparison-all-body")).toContainText("Models ahead");
    await expect(page.locator("#comparison-all-body .comparison-best")).not.toHaveCount(0);
});

test("loaded comparison omits unavailable optional tables without staying in loading state", async ({ page }) => {
    await localAPI(page);
    const response = structuredClone(comparison);
    delete response.subsets.weekly_reference_top24;
    delete response.weekly_ranking;
    let release;
    const pending = new Promise((resolve) => { release = resolve; });
    await page.route("**/api/comparison", async (route) => {
        await pending;
        await route.fulfill({
            contentType: "application/json",
            headers: { "X-FFP-Contract-Version": "1.0" },
            body: JSON.stringify(response),
        });
    });
    await page.goto("/#comparison");
    for (const id of ["comparison-weekly-top24", "comparison-weekly-capture"]) {
        await expect(page.locator(`#${id}`)).toContainText("Loading comparison");
    }
    release();
    for (const id of ["comparison-weekly-top24", "comparison-weekly-capture"]) {
        await expect(page.locator(`#${id} .arch-loading`)).toHaveCount(0);
        await expect(page.locator(`#${id} tr`)).toHaveCount(6);
        await expect(page.locator(`#${id} .comparison-empty`)).not.toHaveCount(0);
    }
    await expect(page.locator("#comparison-all-body tr")).toHaveCount(6);
    await expect(page.locator("#comparison-depth-starters tr")).toHaveCount(6);
    await expect(page.locator("#comparison-weekly-consensus")).toHaveCount(0);
});

test("a statistical tie highlights no cell", async ({ page }) => {
    await localAPI(page);
    const response = structuredClone(comparison);
    for (const cohort of Object.values(response.coverage)) {
        for (const cell of Object.values(cohort)) {
            if (cell.uncertainty?.status === "available") cell.uncertainty.winner = "tie";
        }
    }
    await page.route("**/api/comparison", (route) => route.fulfill({
        contentType: "application/json",
        headers: { "X-FFP-Contract-Version": "1.0" },
        body: JSON.stringify(response),
    }));
    await page.goto("/#comparison");
    await expect(page.locator("#comparison-all-body")).toContainText("≈ tie");
    await expect(page.locator("#view-comparison .comparison-best")).toHaveCount(0);
});

for (const [experts, names] of [
    [["espn"], "ESPN"],
    [["nflcom", "rotowire"], "NFL.com and RotoWire"],
    [["nflcom", "rotowire", "custom"], "NFL.com, RotoWire and custom"],
]) {
    test(`timeline renders the supplied expert names: ${names}`, async ({ page }) => {
        await localAPI(page);
        const models = ["ridge", "nn", "attn_nn", "lgbm"];
        const sources = [...models, ...experts];
        await page.route("**/api/timeline?*", (route) => route.fulfill({
            contentType: "application/json",
            headers: { "X-FFP-Contract-Version": "1.0" },
            body: JSON.stringify({
                schema_version: 2, season: 2025, positions: ["QB"], experts, sources,
                model_labels: { ridge: "Ridge", nn: "Neural Net", attn_nn: "Attention NN", lgbm: "LightGBM", nflcom: "NFL.com", rotowire: "RotoWire", espn: "ESPN" },
                weekly: [], releases: [], scoring_components: { QB: ["passing_yards"] }, excluded_sources: {},
                summary: {
                    n: 24, cohort_n: 24, actual_n: 24, evaluated_weeks: 1, total_weeks: 1,
                    source_n: Object.fromEntries(sources.map((source) => [source, 24])),
                    models: Object.fromEntries(models.map((model) => [model, { mae: 2, beat_experts: 1, evaluated_weeks: 1 }])),
                },
            }),
        }));
        await page.goto("/#timeline");
        await expect(page.locator("#view-timeline")).toContainText(`${names} · 24 common player-weeks`);
        await expect(page.locator(".timeline-track-card")).toContainText(`Beat ${experts.length > 1 ? "every expert" : names}: 1 / 1 weeks`);
        if (experts.includes("custom")) await expect(page.locator("#view-timeline")).toContainText("custom 24");
    });
}

test("unsupported contract versions surface a failure instead of fabricated rows", async ({ page }) => {
    await localAPI(page, { version: "2.0" });
    await page.goto("/#predictions");
    await expect(page.locator("#predictions-body")).toContainText("Failed to load predictions");
    await expect(page.locator('[data-player-id="fixture-QB"]')).toHaveCount(0);
});

test("architecture displays the served bundle recipe and labels fallback positions", async ({ page }) => {
    await localAPI(page);
    await page.goto("/#model-architecture");
    const row = page.locator("#arch-config-table tbody tr").filter({ has: page.locator(".arch-pos-cell", { hasText: /^QB$/ }) });
    await expect(row).toContainText("Served nn bundle");
    await expect(row).toContainText("[7]");
    await expect(row).toContainText("2.0e-3");
    await expect(page.locator("#arch-config-table")).toContainText("Configured fallback");
    const accordion = page.locator(".arch-accordion").filter({ has: page.locator(".arch-pos-label", { hasText: /^QB$/ }) });
    await accordion.locator("summary").click();
    await expect(accordion).toContainText("fixture_input");
    await expect(accordion).toContainText("Recorded nn bundle");
});
