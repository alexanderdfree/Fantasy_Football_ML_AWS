# FF Predictor — iOS app

A native **SwiftUI iPhone app** for the Fantasy Football ML project. It's a
read-only client of the existing public API (`https://fantasy.alexfree.me`) and
mirrors the web dashboard: live next-week projections, season leaders, model
accuracy, model-vs-expert comparison, architecture, training history, and docs.

The app consumes the same JSON the website does, showing all four models (Ridge
/ NN / Attention NN / LightGBM) side by side across PPR / Half-PPR / Standard
scoring. The serving app also exposes `/privacy` and `/support` pages for App
Store metadata.

## Build & run (on a Mac with Xcode 15+ )

The Xcode project is **generated from `project.yml`** with [XcodeGen](https://github.com/yonaskolb/XcodeGen)
so the whole app is plain text (authored on Linux). It is gitignored — regenerate
any time:

```bash
brew install xcodegen
cd ios
xcodegen generate
open FFPredictor.xcodeproj
```

Then pick an iPhone simulator (iOS 17+) and run. No third-party Swift packages —
just `URLSession`, Swift Charts, and `WKWebView` (system frameworks).

If you don't want XcodeGen: create a new Xcode **App** project (SwiftUI, iOS 17),
delete the template files, drag in `Sources/` (and `Tests/` for the test target),
and set `Resources/Info.plist` as the Info.plist.

### API base URL

Defaults to production (`https://fantasy.alexfree.me`). To point at a local Flask
instance, set `API_BASE_URL` in `Resources/Info.plist` to the host root (e.g.
`http://127.0.0.1:5050`) and run the server (`python -m src.serving.app` from the
repo root — see the project SETUP.md). `NSAllowsLocalNetworking` already permits
loopback cleartext without weakening production TLS. See `Networking/AppConfig.swift`.

## Architecture

- **MVVM**, iOS 17 `@Observable` stores, `async/await` over `URLSession`.
- `Models/` — `Codable` structs for every endpoint (`Enums.swift` holds the
  shared `Position` / `ScoringFormat` / `PredictionModel`, which centralizes the
  three key conventions the API uses for each model).
- `Networking/` — `APIClient` (actor) + typed `Endpoint`s; `URLCache` for response
  caching. `Persistence/SnapshotCache` keeps the last snapshot on disk for an
  instant / offline first paint.
- `Stores/` — one `@Observable` store per screen; `AppState` holds the global
  scoring + model-display toggles (persisted to `UserDefaults`, same keys as the
  web's `localStorage`).
- `Views/` — five-tab `TabView` (Next Week · Leaders · Accuracy · Compare · More),
  each its own `NavigationStack`. Player detail and wiki docs are value routes.
- `DesignSystem/` — the web's CSS tokens as Swift (`FFColor`/`FFRadius`), dark-locked.

Dense desktop tables are reborn as phone-native screens: Leaders is a card list
(client filter/sort/search, no pagination), Comparison/History are grouped
disclosure lists with pill clusters, Architecture is a per-position spec sheet,
and charts use **Swift Charts**.

## Tests

`Tests/DecodingTests.swift` decodes every model against real captured payloads in
`Tests/Fixtures/*.json` (refresh them with `curl https://fantasy.alexfree.me/api/...`).
Run with **⌘U** in Xcode or:

```bash
cd ios
xcodebuild -project FFPredictor.xcodeproj -scheme FFPredictor \
  -destination 'platform=iOS Simulator,name=iPhone 15,OS=17.0' \
  CODE_SIGNING_ALLOWED=NO test
```

This is the layer that's verifiable without the UI.

## App Store v1

The App Store listing metadata is tracked in `AppStoreMetadata.md`.

- App display name: `Alex's Fantasy Predictions`
- Bundle ID: `me.alexfree.ffpredictor`
- Version/build: `1.0` / `1`
- Price: free
- Categories: Sports primary, Entertainment secondary
- Privacy/support URLs: `https://fantasy.alexfree.me/privacy` and
  `https://fantasy.alexfree.me/support`
- App icon: owned, generated from `scripts/generate_app_icon.py`; no team,
  league, player, text, number, or third-party mark appears in the icon.
- Player photos: remote NFL/ESPN headshots are disabled for v1 while content
  rights are unverified; the UI shows app-owned initials/glyph placeholders.
- Analytics: deferred for v1; no analytics SDK, ads, tracking, accounts, IAP, or
  subscriptions.

## Not in v1 (future)

- `NWPathMonitor` offline banner, K/DST in the live Upcoming slate (the API
  itself defers them — skill positions only), iPad layout, widgets, analytics,
  and verified third-party player imagery.
