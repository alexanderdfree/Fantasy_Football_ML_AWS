function decodeAnchor(anchor) {
    if (!anchor) return null;
    try { return decodeURIComponent(anchor); } catch { return anchor; }
}

export function parseWikiHash(hash) {
    if (!hash || !hash.startsWith("#wiki:")) return null;
    const [slug, ...parts] = hash.slice("#wiki:".length).split(":");
    return { slug, anchor: decodeAnchor(parts.join(":")) };
}

export function wikiLinkTarget(href, currentSlug) {
    const routed = parseWikiHash(href);
    if (routed) return routed;
    if (href.startsWith("#") && href.length > 1 && currentSlug) {
        return { slug: currentSlug, anchor: decodeAnchor(href.slice(1)) };
    }
    return null;
}
