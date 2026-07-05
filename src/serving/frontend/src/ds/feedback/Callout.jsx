import React from "react";

/**
 * Callout — the prominent "approach" banner that heads each view: an icon,
 * a colored title, and a description. tone "accent" (green, default) or
 * "secondary" (blue).
 */
export function Callout({ title, children, icon, tone = "accent", className = "", ...rest }) {
  return (
    <div className={["callout", tone, className].filter(Boolean).join(" ")} {...rest}>
      {icon && <div className="callout-icon">{icon}</div>}
      <div>
        {title != null && <div className="callout-title">{title}</div>}
        <div className="callout-desc">{children}</div>
      </div>
    </div>
  );
}
