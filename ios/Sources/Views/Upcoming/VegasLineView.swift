import SwiftUI

/// Compact Vegas strip — implied team total foregrounded (top Week-1 signal),
/// signed spread (favorite green), O/U total, and a share-of-game-total bar.
struct VegasLineView: View {
    let spread: Double?
    let total: Double?
    let impliedTeamTotal: Double?

    var body: some View {
        HStack(spacing: FFSpacing.sm) {
            chip(label: "Impl", value: Fmt.num(impliedTeamTotal), color: FFColor.accent)
            if let spread {
                chip(
                    label: spread <= 0 ? "Fav" : "Dog",
                    value: Fmt.delta(spread),
                    color: spread <= 0 ? FFColor.accent : FFColor.textSecondary
                )
            }
            if let total {
                chip(label: "O/U", value: Fmt.num(total), color: FFColor.textSecondary)
            }
            Spacer(minLength: FFSpacing.sm)
            if let implied = impliedTeamTotal, let total, total > 0 {
                shareBar(fraction: min(max(implied / total, 0), 1))
            }
        }
    }

    private func chip(label: String, value: String, color: Color) -> some View {
        HStack(spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(FFColor.textMuted)
            Text(value).font(.caption.weight(.semibold)).monospacedDigit().foregroundStyle(color)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(FFColor.bgPrimary)
        .clipShape(Capsule())
    }

    private func shareBar(fraction: Double) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule().fill(FFColor.bgPrimary)
                Capsule().fill(FFColor.accentDim)
                    .frame(width: geo.size.width * fraction)
                Capsule().strokeBorder(FFColor.border)
            }
        }
        .frame(width: 56, height: 8)
        .accessibilityLabel("Implied share of game total \(Int(fraction * 100)) percent")
    }
}
