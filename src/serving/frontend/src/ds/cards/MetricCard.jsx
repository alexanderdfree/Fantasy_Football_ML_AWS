import React from "react";

/**
 * MetricCard — bordered panel with a header and a body of label/value rows.
 * Pass `rows` for the standard MAE/RMSE/R² layout, or `children` for custom
 * body content.
 */
export function MetricCard({ title, rows, children, className = "", ...rest }) {
  return (
    <div className={["metric-card", className].filter(Boolean).join(" ")} {...rest}>
      {title != null && <div className="metric-card-header">{title}</div>}
      <div className="metric-card-body">
        {rows
          ? rows.map((r, i) => (
              <div className="metric-row" key={r.label ?? i}>
                <span className="metric-label">{r.label}</span>
                <span className="metric-value">{r.value}</span>
              </div>
            ))
          : children}
      </div>
    </div>
  );
}
