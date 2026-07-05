/* Bundle the React dashboard into the committed runtime artifact.
 *
 * Output: ../static/js/app.js — same path the vanilla app lived at, so Flask,
 * the Dockerfile COPY, and the pinned static-asset tests all stay unchanged.
 * Identifier minification stays OFF: tests/test_app.py greps the served bundle
 * for the literal `combiner/i?img=${m[1]}&w=${size}` (ESPN headshot resize),
 * which must survive bundling byte-for-byte.
 * Chart.js is NOT bundled — it stays the vendored window.Chart global
 * (static/js/vendor/chart.umd.min.js), pinned by test_vendored_chartjs_is_served.
 */
import esbuild from "esbuild";

const watch = process.argv.includes("--watch");

const options = {
    entryPoints: ["src/main.jsx"],
    bundle: true,
    outfile: "../static/js/app.js",
    jsx: "automatic",
    define: { "process.env.NODE_ENV": '"production"' },
    minifyWhitespace: true,
    minifySyntax: true,
    minifyIdentifiers: false,
    target: "es2019",
    logLevel: "info",
    banner: {
        js: "/* Built from src/serving/frontend — DO NOT EDIT. Rebuild: cd src/serving/frontend && npm run build */",
    },
};

if (watch) {
    const ctx = await esbuild.context(options);
    await ctx.watch();
    console.log("watching src/ …");
} else {
    await esbuild.build(options);
}
