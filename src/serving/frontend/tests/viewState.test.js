import assert from "node:assert/strict";
import test from "node:test";
import { createLatestRequest } from "../src/lib/latestRequest.js";
import { meetsMinimumProjection } from "../src/lib/predictionFilters.js";

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
