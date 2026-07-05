import React from "react";

/** MetaBadge — pill-shaped info chip (e.g. "2025 Season", "Week 12"). */
export function MetaBadge({ children, className = "", ...rest }) {
  return (
    <span className={["meta-badge", className].filter(Boolean).join(" ")} {...rest}>
      {children}
    </span>
  );
}
