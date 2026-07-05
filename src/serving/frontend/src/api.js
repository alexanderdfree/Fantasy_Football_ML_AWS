/* Thin fetch layer over the serving JSON API. Endpoints and key names are the
 * fixed contract shared with src/serving/routes.py — do not rename fields here. */

export async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) {
        // Preserve the HTTP status on the thrown error so callers can tell an
        // expected 404 (e.g. /api/player for a rookie with no backtest history)
        // apart from a real server-side failure (500, etc.) instead of
        // conflating every non-ok response into one benign fallback (#1437).
        const err = new Error(`API error: ${resp.status}`);
        err.status = resp.status;
        throw err;
    }
    return resp.json();
}
