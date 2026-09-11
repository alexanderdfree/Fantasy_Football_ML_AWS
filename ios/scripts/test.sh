#!/usr/bin/env bash
# Compile the real app and run hosted XCTest with startup HTTP isolated locally.
set -euo pipefail
cd "$(dirname "$0")/.."
FFP_DERIVED_DATA="${FFP_DERIVED_DATA:-${TMPDIR:-/tmp}/ffp-client-tests}"
FFP_SIMULATOR_ID="${FFP_SIMULATOR_ID:-$(xcrun simctl list devices available -j | python3 -c 'import json,sys; d=json.load(sys.stdin)["devices"]; print(next(v["udid"] for k in sorted(d, reverse=True) if "iOS" in k for v in d[k] if v["name"].startswith("iPhone")))')}"
xcodegen generate
xcodebuild -project FFPredictor.xcodeproj -scheme FFPredictor \
  -destination "platform=iOS Simulator,id=$FFP_SIMULATOR_ID" \
  -derivedDataPath "$FFP_DERIVED_DATA" CODE_SIGNING_ALLOWED=NO build-for-testing
# Only the generated test app is changed; production Info.plist stays untouched.
/usr/libexec/PlistBuddy -c 'Set :API_BASE_URL http://127.0.0.1:9' \
  "$FFP_DERIVED_DATA/Build/Products/Debug-iphonesimulator/FFPredictor.app/Info.plist"
xcodebuild -project FFPredictor.xcodeproj -scheme FFPredictor \
  -destination "platform=iOS Simulator,id=$FFP_SIMULATOR_ID" \
  -derivedDataPath "$FFP_DERIVED_DATA" CODE_SIGNING_ALLOWED=NO test-without-building
