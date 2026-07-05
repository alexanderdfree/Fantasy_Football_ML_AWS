import React from "react";

/**
 * DropdownMenu — a trigger button that opens a popover of checkbox rows
 * (multi-select). Used for "show / hide columns" and any compact multi-select
 * filter. Controlled: `value` is the array of checked item values; toggling a
 * row calls `onChange(nextValueArray)`. Items flagged `disabled` render checked
 * but locked (e.g. a column that can't be hidden). Closes on outside-click /
 * Escape; stays open while toggling rows.
 */
export function DropdownMenu({
  fieldLabel,
  label = "Options",
  items = [],
  value = [],
  onChange,
  align = "left",
  showCount = true,
  panelTitle,
  id,
  className = "",
  ...rest
}) {
  const [open, setOpen] = React.useState(false);
  const ref = React.useRef(null);

  React.useEffect(() => {
    if (!open) return undefined;
    const onDoc = (e) => { if (ref.current && !ref.current.contains(e.target)) setOpen(false); };
    const onKey = (e) => { if (e.key === "Escape") setOpen(false); };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const toggle = (v) => {
    const set = new Set(value);
    if (set.has(v)) set.delete(v); else set.add(v);
    if (onChange) onChange([...set]);
  };

  const count = items.filter((it) => value.includes(it.value)).length;

  const menu = (
    <div className={["dropdown-menu", className].filter(Boolean).join(" ")} ref={ref}>
      <button
        type="button"
        id={id}
        className={["dropdown-trigger", open ? "open" : ""].filter(Boolean).join(" ")}
        aria-haspopup="true"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        {...rest}
      >
        <span className="dropdown-trigger-label">{label}</span>
        {showCount && <span className="dropdown-count">{count}/{items.length}</span>}
        <svg className="dropdown-caret" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
          <path d="M8 11L3 6h10z" />
        </svg>
      </button>
      {open && (
        <div className={["dropdown-panel", align === "right" ? "align-right" : ""].filter(Boolean).join(" ")} role="group">
          {panelTitle && <div className="dropdown-header">{panelTitle}</div>}
          {items.map((it) => {
            const checked = value.includes(it.value);
            return (
              <label key={it.value} className={["dropdown-item", it.disabled ? "is-locked" : ""].filter(Boolean).join(" ")}>
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={it.disabled}
                  onChange={() => !it.disabled && toggle(it.value)}
                />
                <span className="dropdown-check" aria-hidden="true">
                  <svg viewBox="0 0 12 12" fill="none">
                    <path d="M2.5 6.4L4.8 8.8 9.5 3.6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                </span>
                <span className="dropdown-item-label">{it.label}</span>
                {it.disabled && <span className="dropdown-lock">locked</span>}
              </label>
            );
          })}
        </div>
      )}
    </div>
  );

  if (!fieldLabel) return menu;
  return (
    <div className="filter-group">
      <label className="field-label" htmlFor={id}>{fieldLabel}</label>
      {menu}
    </div>
  );
}
