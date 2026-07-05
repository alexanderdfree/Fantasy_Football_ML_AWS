import React from "react";

/**
 * Banner — thin, left-accent-bordered status line. tone: "info" (blue),
 * "warning" (yellow), "success" (green), "danger" (red).
 */
export function Banner({ children, tone = "info", className = "", ...rest }) {
  return (
    <div className={["banner", tone, className].filter(Boolean).join(" ")} role="status" aria-live="polite" {...rest}>
      {children}
    </div>
  );
}
