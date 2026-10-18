# Team logo assets

These 32 transparent, 200 × 200 PNG logos are served locally so team labels,
matchups and DST detail cards do not depend on an external image request.

Source: ESPN's public NFL logo CDN, retrieved 2026-10-18 with the existing
ESPN image combiner: `https://a.espncdn.com/combiner/i?img=/i/teamlogos/nfl/500/{team}.png&w=200`.
The files preserve the returned PNG bytes. NFL/team marks belong to their
respective owners. Filenames follow the dashboard's team codes (`la`, `jax`,
`was`); historical aliases resolve in `frontend/src/components/TeamLabel.jsx`.
