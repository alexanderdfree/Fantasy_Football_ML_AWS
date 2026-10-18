### [FIXED] Dashboard team and DST logos depended on external image requests

**File(s)**: `src/serving/frontend/src/components/TeamLabel.jsx`,
`src/serving/frontend/src/PlayerModal.jsx`, `src/serving/static/img/teams/`;
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
