import assert from "node:assert/strict";
import { test } from "node:test";
import { createUpcomingWeekStore, projectionFreshnessNotice, projectionCoverageNotice } from "../src/lib/upcomingWeek.js";

const NOW = Date.parse("2026-09-10T12:00:00Z");
const good = (extra = {}) => ({
    available: true, generated_at: new Date(NOW).toISOString(), scoring: { ppr: [{ name: "Player" }] },
    ...extra,
});
const response = (data, status = 200) => ({ status, ok: status < 400, json: async () => data });
const flush = () => new Promise((resolve) => setImmediate(resolve));

function setup(responses) {
    const calls = [];
    const timers = new Map();
    const windowTarget = new EventTarget();
    const documentTarget = new EventTarget();
    documentTarget.visibilityState = "visible";
    let timerId = 0;
    const store = createUpcomingWeekStore({
        fetchFn: (url, options) => {
            calls.push({ url, options });
            const next = responses.shift();
            if (next instanceof Error) throw next;
            return typeof next === "function" ? next(options) : next;
        },
        windowTarget, documentTarget, now: () => NOW,
        setTimer: (fn, delay) => { timers.set(++timerId, { fn, delay }); return timerId; },
        clearTimer: (id) => timers.delete(id),
    });
    const unsubscribe = store.subscribe(() => {});
    const tick = async (delay) => {
        const entry = [...timers].find(([, value]) => value.delay === delay);
        assert.ok(entry, `Expected a ${delay}ms timer`);
        timers.delete(entry[0]);
        entry[1].fn();
        await flush();
    };
    return { store, calls, timers, unsubscribe, tick, windowTarget, documentTarget };
}

test("ready results revalidate every minute and preserve original generation times", async () => {
    const first = good();
    const updated = good({ week: 2 });
    const h = setup([response(first), response(updated)]);
    await flush();
    assert.equal(h.store.getSnapshot().data, first);
    assert.equal(h.calls[0].options.cache, "no-store");
    await h.tick(60000);
    assert.equal(h.store.getSnapshot().data, updated);
    assert.equal(h.store.getSnapshot().data.generated_at, first.generated_at);
    h.unsubscribe();
    assert.equal(h.timers.size, 0);
});

test("background polling pauses while hidden and resumes immediately on return", async () => {
    const first = good();
    const updated = good({ week: 2 });
    const h = setup([response(first), response(updated)]);
    await flush();
    h.documentTarget.visibilityState = "hidden";
    h.documentTarget.dispatchEvent(new Event("visibilitychange"));
    await h.tick(60000);
    assert.equal(h.calls.length, 1);
    assert.equal(h.store.getSnapshot().data, first);
    assert.equal(h.timers.size, 0);
    h.documentTarget.visibilityState = "visible";
    h.documentTarget.dispatchEvent(new Event("visibilitychange"));
    await flush();
    assert.equal(h.calls.length, 2);
    assert.equal(h.store.getSnapshot().data, updated);
    assert.ok([...h.timers.values()].some(({ delay }) => delay === 60000));
    h.unsubscribe();
});

test("focus, visible-page return and manual refresh revalidate; hidden-page events do not", async () => {
    const h = setup(Array.from({ length: 4 }, () => response(good())));
    await flush();
    h.windowTarget.dispatchEvent(new Event("focus"));
    await flush();
    h.documentTarget.visibilityState = "hidden";
    h.documentTarget.dispatchEvent(new Event("visibilitychange"));
    await flush();
    assert.equal(h.calls.length, 2);
    h.documentTarget.visibilityState = "visible";
    h.documentTarget.dispatchEvent(new Event("visibilitychange"));
    await flush();
    await h.store.refresh();
    assert.equal(h.calls.length, 4);
    h.unsubscribe();
});

test("a cached result is displayed immediately on remount and still revalidated", async () => {
    let finish;
    const old = good({ week: 1 });
    const next = good({ week: 2 });
    const h = setup([response(old), new Promise((resolve) => { finish = resolve; })]);
    await flush();
    h.unsubscribe();
    const unsubscribe = h.store.subscribe(() => {});
    await flush();
    assert.equal(h.store.getSnapshot().data, old);
    assert.equal(h.calls.length, 2);
    assert.equal(h.store.getSnapshot().refreshing, true);
    finish(response(next));
    await flush();
    assert.equal(h.store.getSnapshot().data, next);
    unsubscribe();
});

test("warming and failed requests retry after 30 seconds and recover automatically", async () => {
    const h = setup([response({}, 503), new Error("offline"), response(good())]);
    await flush();
    assert.equal(h.store.getSnapshot().state, "warming");
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().state, "error");
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().state, "ready");
    assert.equal(h.store.getSnapshot().problem, null);
    h.unsubscribe();
});

test("transient errors and warming preserve last-good rows with a warning", async () => {
    const old = good();
    const h = setup([response(old), response({}, 500), response({}, 503), response(good({ week: 2 }))]);
    await flush();
    await h.store.refresh();
    assert.equal(h.store.getSnapshot().state, "ready");
    assert.equal(h.store.getSnapshot().data, old);
    assert.match(h.store.getSnapshot().problem, /Could not check/);
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().data, old);
    assert.match(h.store.getSnapshot().problem, /not available yet/);
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().data.week, 2);
    assert.equal(h.store.getSnapshot().problem, null);
    h.unsubscribe();
});

test("verified offseason is rechecked, and unavailable data is never labeled offseason", async () => {
    const h = setup([
        response({ available: false, reason: "offseason" }),
        response({ available: false, reason: "no_roster" }), response(good()),
    ]);
    await flush();
    assert.equal(h.store.getSnapshot().state, "offseason");
    await h.tick(60000);
    assert.equal(h.store.getSnapshot().state, "error");
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().state, "ready");
    h.unsubscribe();
});

test("overlapping refresh triggers share one request and unmount ignores its late result", async () => {
    let finish;
    const h = setup([new Promise((resolve) => { finish = resolve; }), response(good({ week: 2 }))]);
    await flush();
    h.windowTarget.dispatchEvent(new Event("focus"));
    h.documentTarget.dispatchEvent(new Event("visibilitychange"));
    h.store.refresh();
    assert.equal(h.calls.length, 1);
    h.unsubscribe();
    assert.equal(h.calls[0].options.signal.aborted, true);
    h.windowTarget.dispatchEvent(new Event("focus"));
    h.documentTarget.dispatchEvent(new Event("visibilitychange"));
    await flush();
    assert.equal(h.calls.length, 1);
    const unsubscribe = h.store.subscribe(() => {});
    await flush();
    finish(response(good({ week: 1 })));
    await flush();
    assert.equal(h.store.getSnapshot().data.week, 2);
    unsubscribe();
    assert.equal(h.timers.size, 0);
});

test("a hung network request is aborted and schedules recovery", async () => {
    const h = setup([(options) => new Promise((resolve, reject) => {
        options.signal.addEventListener("abort", () => reject(new Error("timeout")));
    }), response(good())]);
    await flush();
    await h.tick(20000);
    assert.equal(h.calls[0].options.signal.aborted, true);
    assert.equal(h.store.getSnapshot().state, "error");
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().state, "ready");
    h.unsubscribe();
});

test("malformed success responses are errors and do not discard last-good rows", async () => {
    const old = good();
    const h = setup([response(old), response({ error: "unknown" }), response({ available: true })]);
    await flush();
    await h.store.refresh();
    assert.equal(h.store.getSnapshot().data, old);
    assert.ok(h.store.getSnapshot().problem);
    await h.tick(30000);
    assert.equal(h.store.getSnapshot().data, old);
    assert.ok(h.store.getSnapshot().problem);
    h.unsubscribe();
});

test("freshness advances with artifact age, including legacy payloads and cached server metadata", () => {
    assert.equal(projectionFreshnessNotice(good(), NOW, NOW), null);
    assert.match(projectionFreshnessNotice(good(), NOW + 18000000, NOW + 18000000), /out of date/);
    assert.match(projectionFreshnessNotice(good({ generated_at: null }), NOW, NOW), /could not be verified/);
    assert.match(projectionFreshnessNotice(good({ generated_at: "invalid" }), NOW, NOW), /could not be verified/);
    const fresh = good({ freshness: { status: "fresh", age_seconds: 14399, max_age_seconds: 14400 } });
    assert.equal(projectionFreshnessNotice(fresh, NOW, NOW), null);
    assert.match(projectionFreshnessNotice(fresh, NOW, NOW + 2000), /out of date/);
    assert.match(projectionFreshnessNotice(good({ freshness: { status: "stale", age_seconds: 1 } }), NOW, NOW), /out of date/);
    assert.match(projectionFreshnessNotice(good({ freshness: { status: "unavailable" } }), NOW, NOW), /could not be verified/);
});

test("coverage distinguishes complete, degraded and unverified legacy artifacts", () => {
    assert.equal(projectionCoverageNotice(good({ data_quality: { status: "complete", issues: [] } })), null);
    assert.match(projectionCoverageNotice(good()), /has not been verified/);
    assert.match(projectionCoverageNotice(good({ data_quality: { status: "degraded", issues: [
        { source: "injuries", message: "Injury reports are unavailable." },
    ] } })), /Injury reports are unavailable/);
});

test("a current verified offseason is not an unverifiable projection, but old or invalid confirmations warn", () => {
    const offseason = {
        available: false, reason: "offseason", generated_at: new Date(NOW).toISOString(),
        freshness: { status: "unavailable", age_seconds: 0, max_age_seconds: 14400, reason: "offseason" },
    };
    assert.equal(projectionFreshnessNotice(offseason, NOW, NOW), null);
    assert.match(projectionFreshnessNotice(offseason, NOW, NOW + 14401000), /out of date/);
    assert.match(projectionFreshnessNotice({
        ...offseason,
        freshness: { ...offseason.freshness, age_seconds: null, reason: "missing_timestamp" },
    }, NOW, NOW), /could not be verified/);
    assert.match(projectionFreshnessNotice({
        ...offseason,
        freshness: { ...offseason.freshness, reason: "invalid_timestamp" },
    }, NOW, NOW), /could not be verified/);
});
