import Foundation

/// Number / delta / fantasy-point formatting — ports the helpers in app.js
/// (`fmt`, `fmtDelta`, `pointEquivMultiplier`, `formatTargetMae`) so the iOS
/// numbers read identically to the web dashboard.
enum Fmt {
    /// `fmt(n, d)` — "--" for nil/NaN, else fixed `d` decimals (default 1).
    static func num(_ n: Double?, _ digits: Int = 1) -> String {
        guard let n, !n.isNaN else { return "--" }
        return String(format: "%.\(digits)f", n)
    }

    /// Signed delta string, 1 decimal (e.g. "+2.3", "-1.0").
    static func delta(_ d: Double) -> String {
        (d > 0 ? "+" : "") + num(d, 1)
    }

    /// pred − actual, or nil when either is missing.
    static func errDelta(_ pred: Double?, _ actual: Double?) -> Double? {
        guard let pred, let actual else { return nil }
        return pred - actual
    }

    /// Per-target fantasy-point-equivalent multiplier (display only). Mirrors
    /// app.js BASE_POINT_EQUIVALENT + RECEPTION_WEIGHT.
    static func pointEquivMultiplier(_ targetKey: String, _ scoring: ScoringFormat) -> Double? {
        if targetKey == "receptions" {
            switch scoring {
            case .ppr: return 1.0
            case .halfPPR: return 0.5
            case .standard: return 0.0
            }
        }
        switch targetKey {
        case "passing_tds": return 4.0
        case "rushing_tds", "receiving_tds": return 6.0
        case "interceptions", "fumbles_lost": return 2.0
        default: return nil
        }
    }

    /// "302.00 yds" or "0.40 TDs (2.40 pts)" — raw unit + implied FP delta.
    static func targetMae(_ val: Double?, targetKey: String, unit: String?, scoring: ScoringFormat) -> String {
        guard let val else { return "--" }
        let raw = (unit?.isEmpty == false) ? "\(num(val, 2)) \(unit!)" : num(val, 2)
        if let mult = pointEquivMultiplier(targetKey, scoring) {
            return "\(raw) (\(num(val * mult, 2)) pts)"
        }
        return raw
    }

    /// "m:ss" training-time, "--" when unknown (History tab).
    static func trainingTime(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "--" }
        let total = Int(seconds.rounded())
        return "\(total / 60):" + String(format: "%02d", total % 60)
    }

    /// "2026-06-08T11:05:59" (UTC, no tz) → "2026-06-08 11:05".
    static func historyTimestamp(_ ts: String?) -> String {
        guard let ts, ts.count >= 16 else { return "--" }
        let datePart = String(ts.prefix(10))
        let timePart = String(ts.dropFirst(11).prefix(5))
        return "\(datePart) \(timePart)"
    }

    private static let iso = ISO8601DateFormatter()
    private static let isoFractional: ISO8601DateFormatter = {
        let f = ISO8601DateFormatter()
        f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return f
    }()
    private static let relative = RelativeDateTimeFormatter()

    /// "Updated 8 minutes ago" from an ISO-8601 `generated_at` (with or without
    /// fractional seconds / timezone). `now` is injectable for tests.
    static func relativeTime(fromISO string: String?, now: Date = Date()) -> String? {
        guard let string else { return nil }
        let date = iso.date(from: string)
            ?? isoFractional.date(from: string)
            ?? ISO8601DateFormatter().date(from: string + "Z") // bare "…T…:…" (UTC)
        guard let date else { return nil }
        return relative.localizedString(for: date, relativeTo: now)
    }
}
