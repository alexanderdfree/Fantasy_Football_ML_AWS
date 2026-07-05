/* TeamLabel / MatchupLabel — a team (or matchup) abbreviation followed by the
 * team logo, sized to the surrounding text. Logos are hotlinked from ESPN's
 * public NFL logo CDN (the same CDN-imagery approach as player headshots); a
 * failed logo load hides the <img> so the abbreviation still reads cleanly. */

export function teamLogoUrl(abbr) {
    return `https://a.espncdn.com/i/teamlogos/nfl/500/${String(abbr || "").toLowerCase()}.png`;
}

export function TeamLogo({ abbr }) {
    if (!abbr) return null;
    return (
        <img
            className="team-logo"
            src={teamLogoUrl(abbr)}
            alt=""
            loading="lazy"
            decoding="async"
            onError={(e) => { e.currentTarget.style.display = "none"; }}
        />
    );
}

/* Team abbreviation + its logo (e.g. "BUF" + Bills mark). */
export function TeamLabel({ abbr }) {
    if (!abbr) return null;
    return (
        <span className="team-label">
            <span>{abbr}</span>
            <TeamLogo abbr={abbr} />
        </span>
    );
}

/* "vs"/"@" + opponent abbreviation + opponent logo (e.g. "vs KC" + Chiefs mark). */
export function MatchupLabel({ opponent, isHome }) {
    if (!opponent) return "—";
    return (
        <span className="team-label">
            <span>{`${isHome === 1 ? "vs" : "@"} ${opponent}`}</span>
            <TeamLogo abbr={opponent} />
        </span>
    );
}
