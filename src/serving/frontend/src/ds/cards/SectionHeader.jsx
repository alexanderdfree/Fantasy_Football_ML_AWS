import React from "react";

/** SectionHeader — the 16px/600 label that titles a content block. */
export function SectionHeader({ children, className = "", ...rest }) {
  return (
    <div className={["section-header", className].filter(Boolean).join(" ")} {...rest}>
      {children}
    </div>
  );
}
