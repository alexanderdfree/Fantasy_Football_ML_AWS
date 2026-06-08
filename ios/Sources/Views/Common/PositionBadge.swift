import SwiftUI

/// Colored position capsule (QB red, RB green, …) — text on a 15% tint.
struct PositionBadge: View {
    let position: String
    var size: Font = .caption2

    private var pos: Position? { Position(rawValue: position) }
    private var color: Color { pos?.color ?? FFColor.textSecondary }

    var body: some View {
        Text(position)
            .font(size.weight(.bold))
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .clipShape(Capsule())
            .accessibilityLabel(pos?.fullName ?? position)
    }
}
