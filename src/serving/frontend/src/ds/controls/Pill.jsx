import React from "react";

/**
 * Pill — the dashboard's core filter / toggle button. Rounded, low-contrast
 * by default; green-tinted when `active`; dimmed + non-interactive when
 * `disabled`. Place inside a <PillGroup> for filter rows and segmented toggles.
 */
export function Pill({
  children,
  active = false,
  disabled = false,
  value,
  title,
  onClick,
  className = "",
  ...rest
}) {
  const cls = ["pill", active ? "active" : "", className].filter(Boolean).join(" ");
  return (
    <button
      type="button"
      className={cls}
      disabled={disabled}
      onClick={onClick}
      title={title}
      data-value={value}
      aria-pressed={active}
      {...rest}
    >
      {children}
    </button>
  );
}
