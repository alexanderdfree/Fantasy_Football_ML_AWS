import React from "react";

/**
 * PlayerCell — circular headshot + player name, the Player column's content.
 * Falls back to an empty circle when `src` is absent; DST (a team unit) shows
 * the name only, with no photo.
 */
export function PlayerCell({ name, src, position, className = "", ...rest }) {
  const isDST = String(position || "").toUpperCase() === "DST";
  return (
    <div className={["player-cell", className].filter(Boolean).join(" ")} {...rest}>
      {!isDST &&
        (src ? (
          <img className="player-headshot" src={src} alt="" loading="lazy" decoding="async" />
        ) : (
          <div className="player-headshot" />
        ))}
      <span className="player-name">{name}</span>
    </div>
  );
}
