import React from "react";

/**
 * Modal — centered dialog over a dim scrim. Controlled via `open`; clicking the
 * scrim or the × calls `onClose`. Children render inside the padded panel; use
 * the .modal-header / .modal-headshot / .modal-title helpers for the player card.
 */
export function Modal({ open, onClose, children, maxWidth, className = "", ...rest }) {
  const onScrim = (e) => {
    if (e.target === e.currentTarget && onClose) onClose();
  };
  return (
    <div className={["modal", open ? "open" : "", className].filter(Boolean).join(" ")} onClick={onScrim} role="dialog" aria-modal="true" {...rest}>
      <div className="modal-content" style={maxWidth ? { maxWidth } : undefined}>
        {onClose && (
          <button className="modal-close" onClick={onClose} aria-label="Close">&times;</button>
        )}
        {children}
      </div>
    </div>
  );
}
