import React from "react";

/**
 * DataTable — the dashboard's scrollable, sortable player table. Driven by a
 * `columns` spec and a `rows` array; clicking a sortable header calls `onSort`,
 * clicking a row calls `onRowClick`. Wraps itself in the bordered container.
 *
 * column = { key, label, align?: "left"|"center"|"right", sortable?, width?,
 *            className?, render?: (row, i) => node }
 */
export function DataTable({
  columns = [],
  rows = [],
  sort,
  onSort,
  onRowClick,
  rowKey = "id",
  getRowProps,
  className = "",
}) {
  const alignCls = (a) => (a === "center" ? "cell-center" : a === "right" ? "cell-right" : "");
  const tableCls = ["data-table", onRowClick ? "clickable" : "", className].filter(Boolean).join(" ");

  return (
    <div className="table-container">
      <table className={tableCls}>
        <thead>
          <tr>
            {columns.map((c) => {
              const active = sort && sort.key === c.key;
              const th = [alignCls(c.align), c.sortable ? "sortable" : "", active ? "active-sort" : "", c.className || ""]
                .filter(Boolean)
                .join(" ");
              const arrow = active ? (sort.order === "desc" ? "\u25BC" : "\u25B2") : "";
              return (
                <th
                  key={c.key}
                  className={th}
                  style={c.width ? { width: c.width } : undefined}
                  onClick={c.sortable && onSort ? () => onSort(c.key) : undefined}
                >
                  {c.label}
                  {c.sortable && <span className="sort-arrow">{arrow}</span>}
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const extra = getRowProps ? getRowProps(r, i) : {};
            return (
              <tr key={r[rowKey] ?? i} onClick={onRowClick ? () => onRowClick(r, i) : undefined} {...extra}>
                {columns.map((c) => (
                  <td key={c.key} className={[alignCls(c.align), c.className || ""].filter(Boolean).join(" ")}>
                    {c.render ? c.render(r, i) : r[c.key]}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
