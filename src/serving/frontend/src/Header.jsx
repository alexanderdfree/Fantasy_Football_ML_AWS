/* Global header: glowing wordmark, sun/moon theme toggle, scoring segmented
 * toggle (persisted), debounced player search. Markup mirrors index.html so
 * style.css applies unchanged. */
import { PillGroup } from "./components/common.jsx";

const SCORING_OPTIONS = [
    { value: "ppr", label: "Full PPR" },
    { value: "half_ppr", label: "Half PPR" },
    { value: "standard", label: "Standard" },
];

function SunIcon() {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <circle cx="12" cy="12" r="4" />
            <path d="M12 2v2" /><path d="M12 20v2" /><path d="m4.93 4.93 1.41 1.41" /><path d="m17.66 17.66 1.41 1.41" />
            <path d="M2 12h2" /><path d="M20 12h2" /><path d="m6.34 17.66-1.41 1.41" /><path d="m19.07 4.93-1.41 1.41" />
        </svg>
    );
}

function MoonIcon() {
    return (
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z" />
        </svg>
    );
}

export function Header({ theme, onToggleTheme, scoring, onScoring, search, onSearch }) {
    const oled = theme === "oled";
    return (
        <header className="header">
            <div className="header-left">
                <h1 className="logo">Fantasy Football Predictor</h1>
                <span className="subtitle">Position-Specific ML Predictions</span>
            </div>
            <div className="header-right">
                <button
                    type="button"
                    className="theme-toggle"
                    id="theme-toggle"
                    aria-label="Toggle color theme"
                    aria-pressed={oled}
                    title={oled ? "OLED night mode — switch to midnight" : "Midnight mode — switch to OLED night"}
                    onClick={onToggleTheme}
                >
                    <span className={`theme-ico${oled ? "" : " is-active"}`}><SunIcon /></span>
                    <span className="theme-divider" aria-hidden="true">/</span>
                    <span className={`theme-ico${oled ? " is-active" : ""}`}><MoonIcon /></span>
                </button>
                <PillGroup
                    className="scoring-toggle pill-group"
                    id="scoring-filter"
                    options={SCORING_OPTIONS}
                    value={scoring}
                    onChange={onScoring}
                />
                <div className="search-box">
                    <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
                    </svg>
                    <input
                        type="text"
                        id="search-input"
                        placeholder="Search players..."
                        value={search}
                        onChange={(e) => onSearch(e.target.value)}
                    />
                </div>
            </div>
        </header>
    );
}
