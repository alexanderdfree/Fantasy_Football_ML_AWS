import assert from "node:assert/strict";
import test from "node:test";
import { createLatestRequest } from "../src/lib/latestRequest.js";
import { meetsMinimumProjection, sliceAccuracy } from "../src/lib/predictionFilters.js";
import { parseWikiHash, wikiLinkTarget } from "../src/lib/wikiLinks.js";

function deferred() {
    let resolve, reject;
    const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
    return { promise, resolve, reject };
}

for (const staleResult of ["success", "error"]) {
    test(`later navigation survives an earlier ${staleResult}`, async () => {
        const request = createLatestRequest();
        const older = deferred();
        const newer = deferred();
        const visible = [];
        const onSuccess = (value) => visible.push(value);
        const onError = (error) => visible.push(error.message);
        const oldRun = request.run(() => older.promise, onSuccess, onError);
        const newRun = request.run(() => newer.promise, onSuccess, onError);
        newer.resolve("new page");
        await newRun;
        if (staleResult === "success") older.resolve("old page");
        else older.reject(new Error("old page error"));
        await oldRun;
        assert.deepEqual(visible, ["new page"]);
    });

    test(`unmount discards a pending ${staleResult}`, async () => {
        const request = createLatestRequest();
        const pending = deferred();
        const updates = [];
        const run = request.run(() => pending.promise, (x) => updates.push(x), (x) => updates.push(x));
        request.cancel();
        if (staleResult === "success") pending.resolve("detached page");
        else pending.reject(new Error("detached error"));
        await run;
        assert.deepEqual(updates, []);
    });
}

test("cached navigation supersedes an in-flight page and current failures remain visible", async () => {
    const request = createLatestRequest();
    const pending = deferred();
    const updates = [];
    const success = (value) => updates.push(value);
    const failure = (error) => updates.push(error.message);
    const oldRun = request.run(() => pending.promise, success, failure);
    await request.run(() => "cached page", success, failure);
    pending.resolve("old page");
    await oldRun;
    await request.run(() => Promise.reject(new Error("current error")), success, failure);
    assert.deepEqual(updates, ["cached page", "current error"]);
});

test("minimum points includes the kicker's Ridge projection", () => {
    const kicker = { position: "K", ridge_pred: 10, nn_pred: 2, attn_nn_pred: 2, lgbm_pred: 2 };
    assert.equal(meetsMinimumProjection(kicker, 8), true);
    assert.equal(meetsMinimumProjection({ ...kicker, ridge_pred: 7 }, 8), false);
    assert.equal(meetsMinimumProjection({ position: "K", ridge_pred: 8 }, 8), true);
    assert.equal(meetsMinimumProjection({ position: "K" }, 8), false);
    assert.equal(meetsMinimumProjection({ position: "K" }, NaN), true);
    assert.equal(meetsMinimumProjection({ espn_pred: 9 }, 8), true);
});

const accuracySources = [{ key: "ridge_pred", label: "Ridge" }, { key: "nn_pred", label: "NN" }];

test("local Wiki headings retain their document route while other links keep their destination", () => {
    assert.deepEqual(wikiLinkTarget("#1-context", "architecture"),
        { slug: "architecture", anchor: "1-context" });
    assert.deepEqual(wikiLinkTarget("#wiki:batch-design:overview", "architecture"),
        { slug: "batch-design", anchor: "overview" });
    assert.deepEqual(wikiLinkTarget("#wiki:batch-design", "architecture"),
        { slug: "batch-design", anchor: null });
    assert.equal(wikiLinkTarget("https://example.com/#heading", "architecture"), null);
    assert.equal(wikiLinkTarget("#", "architecture"), null);
});

test("Wiki deep-link reloads resolve encoded and literal heading IDs consistently", () => {
    const expected = { slug: "architecture", anchor: "some heading:part" };
    assert.deepEqual(wikiLinkTarget("#some%20heading%3Apart", "architecture"), expected);
    assert.deepEqual(parseWikiHash("#wiki:architecture:some%20heading:part"), expected);
    assert.deepEqual(parseWikiHash("#wiki:architecture:100%"),
        { slug: "architecture", anchor: "100%" });
});
const comparisonRow = (extra = {}) => ({
    actual: 16, comparison_actual: 10,
    comparison_actual_basis: "shared_projected_components_v1", ...extra,
});

test("slice winner uses one common sample and preserves complete-coverage controls", () => {
    const sparse = [comparisonRow({ ridge_pred: 11, nn_pred: 10.5 }),
        comparisonRow({ ridge_pred: null, nn_pred: 30 })];
    assert.deepEqual(sliceAccuracy(sparse, accuracySources), {
        n: 1, cohortN: 2, best: { label: "NN", mae: 0.5 },
    });
    const full = sparse.map((row) => ({ ...row, ridge_pred: 11, nn_pred: 10.5 }));
    assert.deepEqual(sliceAccuracy(full, accuracySources), {
        n: 2, cohortN: 2, best: { label: "NN", mae: 0.5 },
    });
});

test("unprojected actuals cannot change a comparison winner", () => {
    const rows = [comparisonRow({ ridge_pred: 10, nn_pred: 16 })];
    assert.deepEqual(sliceAccuracy(rows, accuracySources).best, { label: "Ridge", mae: 0 });
    assert.deepEqual(sliceAccuracy(rows.map((row) => ({ ...row, actual: 100 })), accuracySources).best,
        { label: "Ridge", mae: 0 });
});

test("missing legacy truth, unknown components and disjoint forecasts remain unavailable", () => {
    for (const rows of [
        [{ actual: 10, ridge_pred: 10, nn_pred: 11 }],
        [comparisonRow({ comparison_actual: null, ridge_pred: 10, nn_pred: 11 })],
        [comparisonRow({ ridge_pred: 10 }), comparisonRow({ nn_pred: 11 })],
        [comparisonRow({ comparison_actual: Infinity, ridge_pred: 10, nn_pred: 11 })],
    ]) {
        assert.equal(sliceAccuracy(rows, accuracySources).best, null);
        assert.equal(sliceAccuracy(rows, accuracySources).n, 0);
    }
});

test("zero forecasts stay comparable and excluded kicker sources cannot win", () => {
    const sources = [...accuracySources, { key: "nflcom_pred", label: "NFL.com" }];
    const rows = [comparisonRow({ comparison_actual: 0, ridge_pred: 0, nn_pred: 1,
        nflcom_pred: 0, comparison_excluded_sources: ["nflcom"] })];
    assert.deepEqual(sliceAccuracy(rows, sources), { n: 1, cohortN: 1, best: { label: "Ridge", mae: 0 } });
});
