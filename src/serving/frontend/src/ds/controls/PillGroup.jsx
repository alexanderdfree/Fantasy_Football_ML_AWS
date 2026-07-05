import React from "react";

/**
 * PillGroup — flex container for a row of <Pill>s.
 *   variant="filter"    → standalone rounded pills with a small gap (default)
 *   variant="segmented" → connected scoring toggle in a bordered track
 */
export function PillGroup({ children, variant = "filter", className = "", ...rest }) {
  const cls = ["pill-group", variant === "segmented" ? "segmented" : "", className]
    .filter(Boolean)
    .join(" ");
  return (
    <div className={cls} role="group" {...rest}>
      {children}
    </div>
  );
}
