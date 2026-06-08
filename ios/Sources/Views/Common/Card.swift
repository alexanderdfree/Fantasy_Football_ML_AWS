import SwiftUI

/// Rounded card container matching the web `.card` (bg-card + border + radius).
struct Card<Content: View>: View {
    var padding: CGFloat = FFSpacing.lg
    @ViewBuilder var content: Content

    var body: some View {
        content
            .padding(padding)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(FFColor.bgCard)
            .clipShape(RoundedRectangle(cornerRadius: FFRadius.lg, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: FFRadius.lg, style: .continuous)
                    .strokeBorder(FFColor.border)
            )
    }
}

/// Small uppercase section label (the web's 11px tracked caption).
struct SectionLabel: View {
    let text: String
    var body: some View {
        Text(text.uppercased())
            .font(.caption2.weight(.semibold))
            .tracking(0.5)
            .foregroundStyle(FFColor.textMuted)
    }
}
