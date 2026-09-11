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
    await expect(page.locator("table").filter({ has: page.locator("#comparison-all-body") }).getByRole("columnheader", { name: "ESPN" })).toBeVisible();
});

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
