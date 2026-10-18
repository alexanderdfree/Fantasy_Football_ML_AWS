import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";

const fixtures = JSON.parse(readFileSync(new URL("../../../../ios/Tests/Fixtures/client_contract.json", import.meta.url)));
const comparison = JSON.parse(readFileSync(new URL("../../../../ios/Tests/Fixtures/comparison_current.json", import.meta.url)));
const playerName = "Marquez Valdes-Scantling";
const longFeature = "prior_season_mean_total_fantasy_points_exp";
const wikiPath = "docs/archive/design_weather_and_odds.md";
const models = ["ridge", "nn", "attn_nn", "lgbm"];
const modelLabels = { ridge: "Ridge", nn: "Neural Net", attn_nn: "Attention NN", lgbm: "LightGBM", espn: "ESPN" };

async function localAPI(page) {
    const snapshot = structuredClone(fixtures.snapshot);
    const architecture = structuredClone(fixtures.architecture);
    architecture.positions.QB.features.other = [...(architecture.positions.QB.features.other || []), longFeature];
    for (const rows of Object.values(snapshot.scoring)) {
        rows.forEach((row, index) => {
            row.team = index % 2 ? "BUF" : "KC";
            row.age = 24 + index;
            if (row.position === "WR") {
                row.name = playerName;
                row.headshot = "/mobile-headshot.svg";
            }
        });
    }
    const sources = [...models, "espn"];
    const mae = Object.fromEntries(sources.map(source => [source, 2]));
    const sourceN = Object.fromEntries(sources.map(source => [source, 24]));
    const historyRows = Array.from({ length: 12 }, (_, index) => ({
        run_id: `mobile-${index}`, timestamp: `2026-09-${String(18 - index).padStart(2, "0")}T12:00:00`,
        git_hash: `abcdef${index}`, positions: ["QB"], total_elapsed_sec: 120,
        ...Object.fromEntries(models.map(model => [model, [{ position: "QB", mae: 2, rmse: 3 }]])),
    }));

    // The committed bundle, vendored charts, CSS, and every API reply stay local.
    await page.route("**/*", async route => {
        const url = new URL(route.request().url());
        if (url.origin !== "http://127.0.0.1:4173") return route.abort();
        if (url.pathname === "/mobile-headshot.svg") {
            return route.fulfill({ contentType: "image/svg+xml", body: '<svg xmlns="http://www.w3.org/2000/svg" width="88" height="88"><circle cx="44" cy="44" r="40" fill="#3d8bff"/></svg>' });
        }
        if (!url.pathname.startsWith("/api/")) return route.continue();
        const scoring = url.searchParams.get("scoring") || "ppr";
        let body = {};
        if (url.pathname === "/api/snapshot") body = snapshot;
        else if (url.pathname === "/api/predictions") body = { ...fixtures.predictions[scoring], players: snapshot.scoring[scoring] };
        else if (url.pathname === "/api/weeks") body = { weeks: [1] };
        else if (url.pathname === "/api/teams") body = { teams: ["BUF", "KC"] };
        else if (url.pathname === "/api/upcoming_week") body = {
            available: true, week_label: "2026 Week 7", positions: ["QB", "RB", "WR", "TE", "K", "DST"],
            scoring: snapshot.scoring, generated_at: new Date().toISOString(),
        };
        else if (url.pathname === "/api/comparison") body = comparison;
        else if (url.pathname === "/api/model_architecture") body = architecture;
        else if (url.pathname === "/api/metrics") body = Object.fromEntries(
            ["Ridge Regression", "Neural Network", "Attention NN", "LightGBM"].map(label => [label, {
                overall: { mae: 2, rmse: 3, r2: 0.7 }, by_position: [{ position: "QB", mae: 2, r2: 0.7 }],
            }]),
        );
        else if (url.pathname === "/api/weekly_accuracy") body = { weeks: [1], ridge_mae: [2], nn_mae: [2], attn_nn_mae: [2], lgbm_mae: [2] };
        else if (url.pathname === "/api/position_details") body = { QB: {
            label: "Quarterback", n_features: 10, n_samples_test: 24,
            targets: [{ key: "passing_yards", label: "Passing yards", formula: "yards / 25" }],
            architecture: { backbone: [128, 64], head_hidden: 16 }, specific_features: ["passing_yards"],
        } };
        else if (url.pathname === "/api/benchmark_history") body = { repo_slug: "owner/repo", rows: historyRows };
        else if (url.pathname === "/api/timeline") body = {
            schema_version: 2, season: 2025, positions: ["QB"], experts: ["espn"], sources, model_labels: modelLabels,
            scoring_components: { QB: ["passing_yards"] }, excluded_sources: {}, releases: [],
            weekly: [{ week: 1, n: 24, cohort_n: 24, actual_n: 24, source_n: sourceN, mae, edges: mae }],
            summary: { n: 24, cohort_n: 24, actual_n: 24, source_n: sourceN, evaluated_weeks: 1, total_weeks: 1,
                models: Object.fromEntries(models.map(model => [model, { mae: 2, beat_experts: 0, evaluated_weeks: 1 }])) },
        };
        else if (url.pathname === "/api/wiki/index") body = [{ group: "Documentation", name: "Architecture", slug: "architecture" }];
        else if (url.pathname === "/api/wiki/architecture") body = { html: `<h1>Architecture</h1><p>Mobile documentation fixture.</p><ul><li><a id="mobile-prose-link" href="${wikiPath}">${wikiPath}</a></li><li><code id="mobile-inline-code">${wikiPath}</code></li></ul><pre><code>${"wide_code_column ".repeat(20)}</code></pre>` };
        else if (url.pathname.startsWith("/api/player/")) body = { ...fixtures.player[scoring], name: playerName, headshot: "/mobile-headshot.svg" };
        return route.fulfill({ contentType: "application/json", headers: { "X-FFP-Contract-Version": "1.0" }, body: JSON.stringify(body) });
    });
}

async function fitsHorizontally(locator, width) {
    const box = await locator.boundingBox();
    expect(box).not.toBeNull();
    expect(box.x).toBeGreaterThanOrEqual(-1);
    expect(box.x + box.width).toBeLessThanOrEqual(width + 1);
}

async function documentFits(page) {
    await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth - innerWidth)).toBeLessThanOrEqual(0);
}

for (const width of [320, 375, 390, 768, 820, 1024, 1440]) {
    test(`dashboard controls and populated views fit ${width}px`, async ({ page }) => {
        await page.setViewportSize({ width, height: 900 });
        await localAPI(page);
        await page.goto("/?mobile-layout=predictions#predictions");
        await expect(page.locator("#predictions-body > tr")).toHaveCount(6);
        await documentFits(page);
        if (width <= 390) expect((await page.locator(".header").boundingBox()).height).toBeLessThan(170);

        const positions = page.locator("#position-filter");
        for (const label of ["All", "QB", "RB", "WR", "TE", "FLEX", "K", "DST"]) {
            const button = positions.getByRole("button", { name: label, exact: true });
            await expect(button).toBeInViewport();
            await fitsHorizontally(button, width);
        }
        await positions.getByRole("button", { name: "DST", exact: true }).click();
        await expect(page.locator("#predictions-body > tr")).toHaveCount(1);
        await expect(page.locator("#predictions-body .col-pos")).toHaveText("DST");
        await positions.getByRole("button", { name: "FLEX", exact: true }).click();
        await expect(page.locator("#predictions-body > tr")).toHaveCount(3);
        await positions.getByRole("button", { name: "All", exact: true }).click();

        const liveRow = page.locator(".filters-row:not(.filters-row--measure)");
        await fitsHorizontally(liveRow.locator(".meta-badge"), width);
        const columns = page.locator("#column-filter-button");
        const filters = liveRow.getByRole("button", { name: /^Filters/ });
        await fitsHorizontally(columns, width);
        await fitsHorizontally(filters, width);
        await columns.click();
        let panel = liveRow.locator(".dropdown-panel");
        await fitsHorizontally(panel, width);
        await panel.getByText("Team", { exact: true }).click();
        await expect(page.locator("#predictions-head .col-team")).toHaveCount(0);
        await panel.getByText("Team", { exact: true }).click();
        await expect(page.locator("#predictions-head .col-team")).toHaveCount(1);
        await columns.click();

        await filters.click();
        panel = liveRow.locator(".dropdown-panel");
        await fitsHorizontally(panel, width);
        if (await panel.getByRole("checkbox", { name: "Team", exact: true }).isChecked()) {
            await panel.getByText("Team", { exact: true }).click();
            await expect(page.locator("#team-filter")).toHaveCount(0);
        }
        await panel.getByText("Team", { exact: true }).click();
        await filters.click();
        await page.locator("#team-filter").selectOption("BUF");
        await expect(page.locator("#predictions-body > tr")).toHaveCount(3);
        await documentFits(page);

        const table = page.locator(".table-container").first();
        await fitsHorizontally(table, width);
        if (width <= 820) {
            expect(await table.evaluate(el => {
                el.scrollLeft = el.scrollWidth;
                return el.scrollWidth > el.clientWidth && el.scrollLeft > 0;
            })).toBe(true);
            await documentFits(page);
        }

        // Distinct queries force document navigation; hash-only goto does not remount App.
        for (const [view, ready] of [
            ["homepage", "#homepage-body > tr"], ["model-performance", "#pos-model-detail .pos-model-card"],
            ["comparison", "#comparison-all-body > tr"], ["model-architecture", "#arch-config-table tbody tr"],
            ["wiki", "#wiki-content h1"], ["history", "#history-body > tr"], ["timeline", "#timeline-table tbody tr"],
        ]) {
            await page.goto(`/?mobile-layout=${view}#${view}`);
            await expect(page.locator(ready).first()).toBeVisible();
            if (view === "model-architecture") {
                const accordion = page.locator(".arch-accordion").filter({ has: page.locator(".arch-pos-label", { hasText: /^QB$/ }) });
                await accordion.locator("summary").click();
                const chip = accordion.locator(".feature-chip").filter({ hasText: longFeature });
                await chip.scrollIntoViewIfNeeded();
                await expect(chip).toBeInViewport({ ratio: 1 });
                await fitsHorizontally(chip, width);
                expect(await chip.evaluate(el => {
                    const chip = el.getBoundingClientRect();
                    const body = el.closest(".arch-accordion-body").getBoundingClientRect();
                    return chip.left >= body.left && chip.right <= body.right && el.scrollWidth <= el.clientWidth;
                })).toBe(true);
            }
            if (view === "wiki") {
                await fitsHorizontally(page.locator("#mobile-prose-link"), width);
                await fitsHorizontally(page.locator("#mobile-inline-code"), width);
                const codeBlock = page.locator("#wiki-content pre");
                await fitsHorizontally(codeBlock, width);
                expect(await codeBlock.evaluate(el => {
                    el.scrollLeft = el.scrollWidth;
                    return el.scrollWidth > el.clientWidth && el.scrollLeft > 0;
                })).toBe(true);
            }
            await documentFits(page);
        }
    });
}

test("mobile deep links and back reveal the active tab without moving the page", async ({ page }) => {
    await page.setViewportSize({ width: 375, height: 650 });
    await localAPI(page);
    for (const view of ["timeline", "history"]) {
        await page.goto(`/?mobile-nav=${view}#${view}`);
        await expect(page.locator(`#view-${view}`)).toBeVisible();
        await expect.poll(async () => {
            const box = await page.locator(`.tab.active[data-view="${view}"]`).boundingBox();
            return box && box.x >= -1 && box.x + box.width <= 376;
        }).toBe(true);
    }
    // Set this before pushState: scroll restoration belongs to each history entry.
    await page.evaluate(() => { history.scrollRestoration = "manual"; });
    await page.locator('.tab[data-view="timeline"]').click();
    await expect(page.locator("#timeline-table tbody tr")).toHaveCount(1);
    await page.evaluate(() => {
        window.scrollTo(0, 250);
        document.querySelector(".nav-tabs").scrollLeft = 0;
    });
    const scrollY = await page.evaluate(() => window.scrollY);
    expect(scrollY).toBeGreaterThan(0);
    await page.goBack();
    await expect(page.locator("#history-body > tr")).toHaveCount(12);
    await expect.poll(async () => {
        const box = await page.locator('.tab.active[data-view="history"]').boundingBox();
        return box && box.x >= -1 && box.x + box.width <= 376;
    }).toBe(true);
    expect(await page.evaluate(() => window.scrollY)).toBe(scrollY);
});

test("phone header controls and long-name player detail remain usable", async ({ page }) => {
    await page.setViewportSize({ width: 320, height: 700 });
    await localAPI(page);
    await page.goto("/#predictions");
    await expect(page.locator("#predictions-body > tr")).toHaveCount(6);
    await page.locator("#search-input").fill(playerName);
    await expect(page.locator("#predictions-body > tr")).toHaveCount(1);
    await page.locator("#scoring-filter").getByRole("button", { name: "Standard", exact: true }).click();
    await expect(page.locator("#predictions-body .col-actual")).toHaveText("12.0");
    await page.locator("#theme-toggle").click();
    await expect(page.locator("#theme-toggle")).toHaveAttribute("aria-pressed", "true");
    await page.locator("#predictions-body .col-player").click();
    const modal = page.locator("#player-modal .modal-content");
    await expect(page.locator("#modal-name")).toHaveText(playerName);
    await expect(page.locator("#modal-headshot")).toBeVisible();
    await expect(page.locator("#player-chart")).toBeInViewport();
    for (const locator of [modal, page.locator("#modal-name"), page.locator("#modal-close"), page.locator("#player-chart")]) {
        await fitsHorizontally(locator, 320);
    }
    expect(await modal.evaluate(el => el.scrollWidth <= el.clientWidth)).toBe(true);
    await page.locator("#modal-close").click();
    await expect(page.locator("#player-modal")).toHaveCount(0);
    await documentFits(page);
});
