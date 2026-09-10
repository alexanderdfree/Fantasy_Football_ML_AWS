const REFRESH_MS = 60 * 1000;
const RETRY_MS = 30 * 1000;
const REQUEST_TIMEOUT_MS = 20 * 1000;
const MAX_AGE_SECONDS = 4 * 60 * 60;

// Keep the last response across tab switches, but always revalidate on return.
// The subscription owns all background work so hidden/unmounted views cannot
// publish an old response into a later visit or leave polling timers behind.
export function createUpcomingWeekStore({
    fetchFn = (...args) => fetch(...args),
    windowTarget = globalThis.window,
    documentTarget = globalThis.document,
    setTimer = setTimeout,
    clearTimer = clearTimeout,
    now = Date.now,
} = {}) {
    let snapshot = { state: "loading", data: null, refreshing: false, problem: null, receivedAt: null };
    const listeners = new Set();
    let retryTimer;
    let requestTimer;
    let controller;
    let pending;
    let generation = 0;

    const publish = (next) => {
        snapshot = next;
        listeners.forEach((listener) => listener());
    };
    const failed = (state, problem) => {
        const keepRows = snapshot.data?.available === true;
        publish({ ...snapshot, state: keepRows ? "ready" : state,
            data: keepRows ? snapshot.data : null, refreshing: false, problem });
    };

    const refresh = () => {
        if (!listeners.size || pending) return pending;
        clearTimer(retryTimer);
        // Resume immediately on visibility/focus instead of polling background tabs.
        if (documentTarget?.visibilityState === "hidden") return;
        const requestGeneration = generation;
        const requestController = new AbortController();
        controller = requestController;
        publish({ ...snapshot, refreshing: true });
        requestTimer = setTimer(() => requestController.abort(), REQUEST_TIMEOUT_MS);
        pending = (async () => {
            let retry = false;
            try {
                const response = await Promise.resolve().then(() => fetchFn("/api/upcoming_week", {
                    cache: "no-store", signal: requestController.signal,
                }));
                if (generation !== requestGeneration) return;
                if (response.status === 503) {
                    retry = true;
                    failed("warming", "The latest projections are not available yet. Retrying automatically.");
                    return;
                }
                if (!response.ok) throw new Error(`API error: ${response.status}`);
                const data = await response.json();
                if (generation !== requestGeneration) return;
                if (!data || typeof data.available !== "boolean"
                    || (data.available && (!data.scoring || typeof data.scoring !== "object"))) {
                    throw new Error("Invalid projection response");
                }
                if (!data.available && data.reason !== "offseason") {
                    retry = true;
                    failed("error", "The latest projections are temporarily unavailable. Retrying automatically.");
                    return;
                }
                publish({ state: data.available ? "ready" : "offseason", data,
                    refreshing: false, problem: null, receivedAt: now() });
            } catch {
                if (generation !== requestGeneration) return;
                retry = true;
                failed("error", "Could not check for updated projections. Retrying automatically.");
            } finally {
                if (generation === requestGeneration) {
                    clearTimer(requestTimer);
                    controller = null;
                    pending = null;
                    retryTimer = setTimer(refresh, retry ? RETRY_MS : REFRESH_MS);
                }
            }
        })();
        return pending;
    };
    const onVisible = () => {
        if (documentTarget?.visibilityState !== "hidden") refresh();
    };

    return {
        getSnapshot: () => snapshot,
        refresh,
        subscribe(listener) {
            listeners.add(listener);
            if (listeners.size === 1) {
                windowTarget?.addEventListener("focus", onVisible);
                documentTarget?.addEventListener("visibilitychange", onVisible);
                refresh();
            }
            return () => {
                listeners.delete(listener);
                if (!listeners.size) {
                    generation += 1;
                    clearTimer(retryTimer);
                    clearTimer(requestTimer);
                    controller?.abort();
                    controller = null;
                    pending = null;
                    snapshot = { ...snapshot, refreshing: false };
                    windowTarget?.removeEventListener("focus", onVisible);
                    documentTarget?.removeEventListener("visibilitychange", onVisible);
                }
            };
        },
    };
}

export const upcomingWeekStore = createUpcomingWeekStore();

export function projectionFreshnessNotice(data, receivedAt, now = Date.now()) {
    if (!data) return null;
    const metadata = data.freshness;
    const generatedAt = Date.parse(data.generated_at);
    const elapsed = receivedAt == null ? 0 : Math.max(0, (now - receivedAt) / 1000);
    const age = Number.isFinite(metadata?.age_seconds)
        ? metadata.age_seconds + elapsed
        : Number.isFinite(generatedAt) ? (now - generatedAt) / 1000 : null;
    const maxAge = metadata?.max_age_seconds > 0 ? metadata.max_age_seconds : MAX_AGE_SECONDS;
    if (metadata?.status === "stale" || (age != null && age > maxAge)) {
        return "These projections are out of date. Injuries, lineups and matchups may have changed.";
    }
    const verifiedOffseason = data.available === false && data.reason === "offseason"
        && metadata?.reason === "offseason";
    if ((metadata?.status === "unavailable" && !verifiedOffseason) || age == null || age < 0) {
        return "Projection freshness could not be verified.";
    }
    return null;
}

export function projectionCoverageNotice(data) {
    if (!data?.available) return null;
    const quality = data.data_quality;
    if (quality?.status === "complete") return null;
    if (quality?.status === "degraded") {
        const messages = [...new Set((quality.issues || []).map((issue) => issue.message).filter(Boolean))];
        return `Some source data is missing or incomplete.${messages.length ? ` ${messages.join(" ")}` : ""}`;
    }
    return "Source coverage has not been verified for this update.";
}
