import React from "react";

/**
 * SearchBox — text input with a leading magnifier icon. Defaults to the
 * product's 280px header width; override via `width`.
 */
export function SearchBox({ value, onChange, placeholder = "Search players...", width, className = "", ...rest }) {
  return (
    <div className={["search-box", className].filter(Boolean).join(" ")} style={width ? { width } : undefined}>
      <svg className="search-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="11" cy="11" r="8" />
        <path d="m21 21-4.35-4.35" />
      </svg>
      <input type="text" value={value} onChange={onChange} placeholder={placeholder} {...rest} />
    </div>
  );
}
