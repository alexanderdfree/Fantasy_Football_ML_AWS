import React from "react";

/**
 * FeatureChip — monospace tag for an engineered feature / config value.
 *   variant="badge" → square (4px), secondary text (the feature-badge style)
 *   variant="pill"  → fully rounded, primary text (the feature-chip style)
 */
export function FeatureChip({ children, variant = "badge", className = "", ...rest }) {
  const base = variant === "pill" ? "feature-chip" : "feature-badge";
  return (
    <span className={[base, className].filter(Boolean).join(" ")} {...rest}>
      {children}
    </span>
  );
}
