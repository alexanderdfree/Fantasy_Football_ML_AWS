/* Thin fetch layer over the serving JSON API. Endpoints and key names are the
 * fixed contract shared with src/serving/routes.py — do not rename fields here. */

export async function fetchJSON(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`API error: ${resp.status}`);
    return resp.json();
}
