import SwiftUI

/// Titled metric tile with a large monospaced value (Player detail, Performance).
struct StatCard: View {
    let title: String
    let value: String
    var accent: Color = FFColor.accent

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title.uppercased())
                .font(.caption2.weight(.semibold))
                .tracking(0.5)
                .foregroundStyle(FFColor.textMuted)
            Text(value)
                .font(.title2.weight(.bold))
                .monospacedDigit()
                .foregroundStyle(accent)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(FFSpacing.md)
        .background(FFColor.bgCard)
        .clipShape(RoundedRectangle(cornerRadius: FFRadius.lg, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: FFRadius.lg, style: .continuous)
                .strokeBorder(FFColor.border)
        )
    }
}
