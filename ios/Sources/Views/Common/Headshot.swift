import SwiftUI

/// App-owned player placeholder. Remote NFL/ESPN player images are intentionally
/// not fetched for the first App Store submission while content rights are
/// unverified.
struct Headshot: View {
    let url: String
    var name: String = ""
    var position: String? = nil
    var size: CGFloat = 36

    var body: some View {
        Circle()
            .fill(
                LinearGradient(
                    colors: [FFColor.bgCard, FFColor.bgSecondary],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
            )
            .overlay(content)
        .frame(width: size, height: size)
        .clipShape(Circle())
        .overlay(Circle().strokeBorder(FFColor.border))
        .accessibilityHidden(true)
    }

    @ViewBuilder
    private var content: some View {
        if position == "DST" {
            Image(systemName: "shield.lefthalf.filled")
                .font(.system(size: size * 0.42, weight: .semibold))
                .foregroundStyle(FFColor.textMuted)
        } else {
            Text(initials)
                .font(.system(size: max(11, size * 0.34), weight: .bold, design: .rounded))
                .foregroundStyle(FFColor.accent)
        }
    }

    private var initials: String {
        let parts = name
            .split(separator: " ")
            .prefix(2)
            .compactMap(\.first)
            .map { String($0).uppercased() }
        if !parts.isEmpty { return parts.joined() }
        return position?.prefix(2).uppercased() ?? "FF"
    }
}
