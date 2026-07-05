import React from "react";

/** Spinner — the brand loading ring (gray track, green head). `size` in px. */
export function Spinner({ size = 40, className = "", style, ...rest }) {
  const s = { width: size, height: size, ...(style || {}) };
  return <div className={["spinner", className].filter(Boolean).join(" ")} style={s} role="status" aria-label="Loading" {...rest} />;
}
