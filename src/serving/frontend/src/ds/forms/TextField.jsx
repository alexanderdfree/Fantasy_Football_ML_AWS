import React from "react";

/**
 * TextField — text or number input in the brand style. Optional uppercase
 * `label` renders above (filter-field layout). Number type defaults to 90px.
 */
export function TextField({ label, type = "text", value, onChange, placeholder, id, className = "", ...rest }) {
  const input = (
    <input
      id={id}
      type={type}
      className={["text-field", label ? "" : className].filter(Boolean).join(" ")}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      {...rest}
    />
  );
  if (!label) return input;
  return (
    <div className={["filter-group", className].filter(Boolean).join(" ")}>
      <label className="field-label" htmlFor={id}>{label}</label>
      {input}
    </div>
  );
}
