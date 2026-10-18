### [FIXED] Dashboard team and DST logos depended on external image requests

**File(s)**: `src/serving/frontend/src/components/TeamLabel.jsx`,
`src/serving/frontend/src/PlayerModal.jsx`,
`src/serving/frontend/src/components/common.jsx`,
`src/serving/static/css/style.css`, `src/serving/static/img/teams/`;
see this record's introducing commit.

**What**: The owner reported broken Saints, Giants and other DST images.
The live URLs and image bytes were valid during investigation, so a persistent
CDN outage or abbreviation defect was not established. Team/matchup labels and
DST detail cards still required external ESPN image requests. A failed label
image was hidden with an inline style that survived a subsequent source change;
the detail card had no failed-image handler.

**Fix**: Bundle all 32 small team logos and use the same local-logo component
for labels, matchups and DST detail cards, including the no-history identity
card. Remount the image when its source changes so an earlier load failure
cannot hide the next team's logo. Preserve displayed team codes and resolve
existing historical aliases to the bundled assets. Browser checks block all
external traffic and verify local image decoding and failure recovery.

**Lesson**: A valid CDN URL during diagnosis does not guarantee an image loads
in the user's browser. Small, stable identity assets can ship with the UI;
verify decoded images with external requests blocked, and reset failed-image
presentation when its source changes.

**Player-column follow-up (2026-10-18)**: After PR #1650, the owner clarified
that the missing image was beside the DST name in the Player column. The live
CAR D/ST row was rank 126 and contained no Player image element. The active
`common.jsx::PlayerCell` explicitly rendered `null` for every DST icon; the
earlier team-column and detail-card checks did not cover that element.
Render the bundled team logo in this shared Player cell, retaining normal
headshot dimensions and containing the complete mark. This restores icons in
both Next Week and Season Leaders. Regression checks now select
`.col-player img.player-headshot` explicitly for all 32 teams and historical
aliases, plus the Panthers, Saints and Giants homepage cases. The corrected
check failed against the previous bundle with 0 of 38 expected Player icons.
Validate the exact reported element before claiming a visual issue is fixed.
