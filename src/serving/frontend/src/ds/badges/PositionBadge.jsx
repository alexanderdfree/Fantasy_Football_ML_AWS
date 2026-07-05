import React from "react";

/**
 * PositionBadge — the colored QB/RB/WR/TE/K/DST chip. Each position carries a
 * fixed hue (red/green/blue/yellow/purple/orange) on a 15% tint.
 */
export function PositionBadge({ position, className = "", ...rest }) {
  const pos = String(position || "").toUpperCase();
  const cls = ["pos-badge", `pos-${pos}`, className].filter(Boolean).join(" ");
  return (
    <span className={cls} {...rest}>
      {pos}
    </span>
  );
}
