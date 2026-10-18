/* TeamLabel / MatchupLabel — a team (or matchup) abbreviation followed by the
 * bundled team logo, sized to the surrounding text. Keep logo rendering
 * independent of third-party CDN availability, including in DST detail cards. */

const TEAMS = new Set("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LA LAC LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split(" "));
const LOGO_ALIASES = { JAC: "JAX", LAR: "LA", OAK: "LV", SD: "LAC", STL: "LA", WSH: "WAS" };

export function teamLogoUrl(abbr) {
    const code = String(abbr || "").trim().toUpperCase();
    const team = LOGO_ALIASES[code] || code;
    return TEAMS.has(team) ? `/static/img/teams/${team.toLowerCase()}.png` : "";
}

export function TeamLogo({ abbr, className = "team-logo", alt = "", ...props }) {
    const src = teamLogoUrl(abbr);
    if (!src) return null;
    return (
        <img
            {...props}
            key={src}
            className={className}
            src={src}
            alt={alt}
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
