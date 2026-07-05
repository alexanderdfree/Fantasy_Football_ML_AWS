/* Theme-aware Chart.js palette. Chart text/grid colors live in the CSS custom
 * properties (which the OLED theme overrides), so read them from the active
 * theme instead of hardcoding hexes. Chart.js is the vendored window.Chart
 * global (static/js/vendor/chart.umd.min.js) — never bundled. */

export function cssVar(name, fallback) {
    const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
}

export function chartTheme() {
    return {
        text: cssVar("--text-secondary", "#9aa0b0"),
        grid: cssVar("--border", "#232d47"),
        heading: cssVar("--text-primary", "#e8eaed"),
        font: cssVar("--font-sans", "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"),
    };
}

export function applyChartTheme() {
    const t = chartTheme();
    if (window.Chart) {
        window.Chart.defaults.color = t.text;
        window.Chart.defaults.borderColor = t.grid;
        window.Chart.defaults.font.family = t.font;
    }
    return t;
}

// Model series hues come from the design-system tokens (--model-*) so CSS and
// charts share a single source. Identical in both themes, so resolved once.
let _colors = null;
export function modelColors() {
    if (!_colors) {
        _colors = {
            ridge: cssVar("--model-ridge", "#3b82f6"),
            nn: cssVar("--model-nn", "#22c55e"),
            attn_nn: cssVar("--model-attn-nn", "#a855f7"),
            lgbm: cssVar("--model-lgbm", "#f59e0b"),
            nflcom: cssVar("--model-nflcom", "#06b6d4"),
            rotowire: cssVar("--model-rotowire", "#ec4899"),
            actual: cssVar("--model-actual", "#e8eaed"),
            ridgeBg: "rgba(59, 130, 246, 0.2)",
            nnBg: "rgba(34, 197, 94, 0.2)",
            attn_nnBg: "rgba(168, 85, 247, 0.2)",
            lgbmBg: "rgba(245, 158, 11, 0.2)",
        };
    }
    return _colors;
}
