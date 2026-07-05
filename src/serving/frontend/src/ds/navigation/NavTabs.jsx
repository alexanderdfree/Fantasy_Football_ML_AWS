import React from "react";
import { Tab } from "./Tab.jsx";

/**
 * NavTabs — the horizontal tab bar. Pass `tabs` ([{ value, label }]) with the
 * active `value` and an `onChange(value)` handler, or compose <Tab>s as children.
 */
export function NavTabs({ tabs, value, onChange, children, className = "", ...rest }) {
  return (
    <nav className={["nav-tabs", className].filter(Boolean).join(" ")} role="tablist" {...rest}>
      {tabs
        ? tabs.map((t) => (
            <Tab key={t.value} active={t.value === value} onClick={() => onChange && onChange(t.value)}>
              {t.label}
            </Tab>
          ))
        : children}
    </nav>
  );
}
