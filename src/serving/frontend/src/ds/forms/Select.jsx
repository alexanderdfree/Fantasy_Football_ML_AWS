import React from "react";

/**
 * Select — the brand dropdown (custom chevron, dark fill). Pass <option>s as
 * children. Optional uppercase `label` renders above as a filter field.
 */
export function Select({ label, value, onChange, children, className = "", id, ...rest }) {
  const select = (
    <select id={id} className={["field-select", label ? "" : className].filter(Boolean).join(" ")} value={value} onChange={onChange} {...rest}>
      {children}
    </select>
  );
  if (!label) return select;
  return (
    <div className={["filter-group", className].filter(Boolean).join(" ")}>
      <label className="field-label" htmlFor={id}>{label}</label>
      {select}
    </div>
  );
}
