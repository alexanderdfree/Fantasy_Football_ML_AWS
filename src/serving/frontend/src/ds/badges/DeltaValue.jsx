import React from "react";

/**
 * DeltaValue — a signed prediction-vs-actual delta, colored by direction.
 * Within ±threshold reads neutral (gray); above reads green (+), below red (−).
 * Mirrors the dashboard's "* Err" columns.
 */
export function DeltaValue({ value, threshold = 1, digits = 1, className = "", ...rest }) {
  if (value == null || isNaN(value)) {
    return <span className={["delta-neutral", className].filter(Boolean).join(" ")} {...rest}>--</span>;
  }
  const n = Number(value);
  const cls = Math.abs(n) < threshold ? "delta-neutral" : n > 0 ? "delta-positive" : "delta-negative";
  const sign = n > 0 ? "+" : "";
  return (
    <span className={[cls, "tabular-nums", className].filter(Boolean).join(" ")} {...rest}>
      {`${sign}${n.toFixed(digits)}`}
    </span>
  );
}
