import React from "react";

/** Tab — a single underline tab. Green text + underline when `active`. */
export function Tab({ children, active = false, onClick, className = "", ...rest }) {
  return (
    <button
      type="button"
      className={["tab", active ? "active" : "", className].filter(Boolean).join(" ")}
      onClick={onClick}
      aria-selected={active}
      role="tab"
      {...rest}
    >
      {children}
    </button>
  );
}
