import React from "react";

/**
 * Pagination — windowed page buttons (« 1 … 5 6 7 … 20 ») matching the
 * dashboard. Renders nothing for a single page. Calls `onChange(page)`.
 */
export function Pagination({ page, totalPages, onChange, maxVisible = 7, className = "" }) {
  if (!totalPages || totalPages <= 1) return null;

  let startPage = Math.max(1, page - Math.floor(maxVisible / 2));
  let endPage = Math.min(totalPages, startPage + maxVisible - 1);
  if (endPage - startPage < maxVisible - 1) startPage = Math.max(1, endPage - maxVisible + 1);

  const go = (p) => () => { if (p >= 1 && p <= totalPages && p !== page) onChange && onChange(p); };
  const btns = [];

  btns.push(
    <button key="prev" className="page-btn" disabled={page === 1} onClick={go(page - 1)} aria-label="Previous">&laquo;</button>
  );
  if (startPage > 1) {
    btns.push(<button key={1} className="page-btn" onClick={go(1)}>1</button>);
    btns.push(<span key="e1" className="page-ellipsis">…</span>);
  }
  for (let p = startPage; p <= endPage; p++) {
    btns.push(
      <button key={p} className={["page-btn", p === page ? "active" : ""].filter(Boolean).join(" ")} onClick={go(p)}>{p}</button>
    );
  }
  if (endPage < totalPages) {
    btns.push(<span key="e2" className="page-ellipsis">…</span>);
    btns.push(<button key={totalPages} className="page-btn" onClick={go(totalPages)}>{totalPages}</button>);
  }
  btns.push(
    <button key="next" className="page-btn" disabled={page === totalPages} onClick={go(page + 1)} aria-label="Next">&raquo;</button>
  );

  return <div className={["pagination", className].filter(Boolean).join(" ")}>{btns}</div>;
}
