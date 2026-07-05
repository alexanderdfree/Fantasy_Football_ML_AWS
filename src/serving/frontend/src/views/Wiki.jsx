/* Wiki — render committed markdown docs in the Wiki tab. The sidebar is
 * fetched once from /api/wiki/index; doc bodies are fetched lazily and
 * cached server-side. Intra-wiki links arrive as `#wiki:slug[:anchor]` and
 * are intercepted to swap content without a page reload. Module-level caches
 * survive unmounts so returning to the tab restores the last page. Wiki
 * sub-page navigation uses replaceState (tab clicks pushState in App) so
 * intra-wiki link clicks don't pile up history entries. */
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { fetchJSON } from "../api.js";

const WIKI_DEFAULT_SLUG = "architecture";
let wikiIndexCache = null; // fetched once from /api/wiki/index
let wikiCurrentSlug = null;
const wikiPageCache = new Map(); // slug → server-sanitized html

function parseWikiHash(hash) {
    if (!hash || !hash.startsWith("#wiki:")) return null;
    const rest = hash.slice("#wiki:".length);
    const [slug, ...anchorParts] = rest.split(":");
    return { slug, anchor: anchorParts.join(":") || null };
}

export function WikiView({ scoring, search, theme, onPlayer, activateView }) {
    const [index, setIndex] = useState(() => (
        wikiIndexCache
            ? { status: "ready", items: wikiIndexCache, message: null }
            : { status: "loading", items: null, message: null }
    ));
    // page: idle (nothing loaded yet) | loading | ready | error. A fresh object
    // per load so the scroll effect re-fires even for a same-slug re-click.
    const [page, setPage] = useState(() => {
        const fromHash = parseWikiHash(location.hash);
        const slug = (fromHash && fromHash.slug) || wikiCurrentSlug || WIKI_DEFAULT_SLUG;
        const html = wikiPageCache.get(slug);
        return html
            ? { status: "ready", slug, html, anchor: fromHash ? fromHash.anchor : null, message: null }
            : { status: "idle", slug: null, html: null, anchor: null, message: null };
    });
    // Sidebar highlight follows the last *successful* load (vanilla parity:
    // it stays on the previous page while a new one loads or errors).
    const [activeSlug, setActiveSlug] = useState(() => wikiCurrentSlug);
    const contentRef = useRef(null);
    const mountedRef = useRef(true);

    useEffect(() => {
        mountedRef.current = true;
        return () => { mountedRef.current = false; };
    }, []);

    const loadWikiPage = useCallback(async (slug, anchor = null) => {
        let html = wikiPageCache.get(slug);
        if (!html) {
            setPage({ status: "loading", slug, html: null, anchor: null, message: null });
            try {
                const data = await fetchJSON(`/api/wiki/${encodeURIComponent(slug)}`);
                if (data.error) throw new Error(data.error);
                wikiPageCache.set(slug, data.html);
                html = data.html;
            } catch (e) {
                console.error("Failed to load wiki page:", e);
                if (mountedRef.current) {
                    setPage({ status: "error", slug, html: null, anchor: null, message: e.message });
                }
                return;
            }
        }
        if (!mountedRef.current) return;
        wikiCurrentSlug = slug;
        setActiveSlug(slug);
        setPage({ status: "ready", slug, html, anchor, message: null });
        const newHash = anchor ? `#wiki:${slug}:${anchor}` : `#wiki:${slug}`;
        if (location.hash !== newHash) {
            history.replaceState(null, "", newHash);
        }
    }, []);

    // On mount: ensure the index is loaded (once, module-cached), then resolve
    // the slug — hash slug → last-viewed slug → default — and load that page.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            if (!wikiIndexCache) {
                try {
                    const items = await fetchJSON("/api/wiki/index");
                    wikiIndexCache = items;
                } catch (e) {
                    console.error("Failed to load wiki index:", e);
                    if (!cancelled) setIndex({ status: "error", items: null, message: e.message });
                    return;
                }
            }
            if (cancelled) return;
            setIndex({ status: "ready", items: wikiIndexCache, message: null });
            const fromHash = parseWikiHash(location.hash);
            const slug = (fromHash && fromHash.slug) || wikiCurrentSlug || WIKI_DEFAULT_SLUG;
            const anchor = fromHash ? fromHash.anchor : null;
            loadWikiPage(slug, anchor);
        })();
        return () => { cancelled = true; };
    }, [loadWikiPage]);

    // Back/forward within the wiki (App keeps this view mounted while the
    // hash still starts with #wiki): re-resolve slug/anchor and load.
    useEffect(() => {
        const onPop = () => {
            if (!(location.hash || "").startsWith("#wiki")) return;
            const parsed = parseWikiHash(location.hash);
            const slug = (parsed && parsed.slug) || wikiCurrentSlug || WIKI_DEFAULT_SLUG;
            loadWikiPage(slug, parsed ? parsed.anchor : null);
        };
        window.addEventListener("popstate", onPop);
        return () => window.removeEventListener("popstate", onPop);
    }, [loadWikiPage]);

    // After a page swap: jump to the anchor if present, else scroll to top.
    useLayoutEffect(() => {
        if (page.status !== "ready") return;
        const contentEl = contentRef.current;
        if (!contentEl) return;
        if (page.anchor) {
            const target = document.getElementById(page.anchor);
            if (target) target.scrollIntoView({ behavior: "auto", block: "start" });
            else contentEl.scrollTop = 0;
        } else {
            contentEl.scrollTop = 0;
        }
    }, [page]);

    // Group sidebar items by .group, preserving the index's insertion order.
    const groups = useMemo(() => {
        const m = new Map();
        (index.items || []).forEach((item) => {
            if (!m.has(item.group)) m.set(item.group, []);
            m.get(item.group).push(item);
        });
        return Array.from(m.entries());
    }, [index.items]);

    const onSidebarClick = (e, slug) => {
        e.preventDefault();
        loadWikiPage(slug);
    };

    // One delegated listener on the content article catches the intra-content
    // `#wiki:` links produced by the server-side link rewriter; other links
    // (external GitHub etc.) pass through.
    const onContentClick = (e) => {
        const a = e.target.closest("a");
        if (!a) return;
        const href = a.getAttribute("href") || "";
        if (!href.startsWith("#wiki:")) return;
        e.preventDefault();
        const parsed = parseWikiHash(href);
        if (parsed) loadWikiPage(parsed.slug, parsed.anchor);
    };

    return (
        <section id="view-wiki" className="view active">
            <div className="wiki-layout">
                <aside className="wiki-sidebar" id="wiki-sidebar" aria-label="Documentation index">
                    {index.status === "loading" && <p className="arch-loading">Loading documentation…</p>}
                    {index.status === "error" && <p className="arch-error">Failed to load index: {index.message}</p>}
                    {index.status === "ready" && groups.map(([group, entries]) => (
                        <div className="wiki-sidebar-group" key={group}>
                            <h3 className="wiki-sidebar-heading">{group}</h3>
                            <ul className="wiki-sidebar-list">
                                {entries.map((item) => (
                                    <li key={item.slug}>
                                        <a
                                            href={`#wiki:${item.slug}`}
                                            className={"wiki-sidebar-link" + (item.slug === activeSlug ? " active" : "")}
                                            data-slug={item.slug}
                                            onClick={(e) => onSidebarClick(e, item.slug)}
                                        >
                                            {item.name}
                                        </a>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    ))}
                </aside>
                {page.status === "ready" ? (
                    <article
                        className="wiki-content"
                        id="wiki-content"
                        ref={contentRef}
                        onClick={onContentClick}
                        /* Server-rendered, bleach-sanitized HTML — safe by contract. */
                        dangerouslySetInnerHTML={{ __html: page.html }}
                    />
                ) : (
                    <article className="wiki-content" id="wiki-content" ref={contentRef}>
                        {page.status === "loading" && <p className="arch-loading">Loading…</p>}
                        {page.status === "error" && <p className="arch-error">Failed to load: {page.message}</p>}
                        {page.status === "idle" && <p className="arch-loading">Select a document.</p>}
                    </article>
                )}
            </div>
        </section>
    );
}
