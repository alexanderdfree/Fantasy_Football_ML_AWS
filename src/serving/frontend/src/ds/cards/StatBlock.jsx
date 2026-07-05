import React from "react";

/**
 * StatBlock — an uppercase micro-label over a large value. Value is accent
 * green by default (the modal's season stats); set `neutral` for white.
 */
export function StatBlock({ label, value, neutral = false, className = "", ...rest }) {
  return (
    <div className={["stat-block", className].filter(Boolean).join(" ")} {...rest}>
      <span className="stat-block-label">{label}</span>
      <span className={["stat-block-value", neutral ? "neutral" : ""].filter(Boolean).join(" ")}>{value}</span>
    </div>
  );
}
