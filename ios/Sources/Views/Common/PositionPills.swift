import SwiftUI

/// Horizontally-scrollable position filter (ALL / QB … DST). `disabled` greys
/// out positions (e.g. K/DST on the Upcoming screen — "coming soon").
struct PositionPills: View {
    @Binding var selection: Position? // nil == ALL
    var includeAll: Bool = true
    var disabled: Set<Position> = []

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: FFSpacing.sm) {
                if includeAll {
                    pill(title: "ALL", active: selection == nil, disabled: false) { selection = nil }
                }
                ForEach(Position.displayOrder) { p in
                    pill(title: p.rawValue, active: selection == p, disabled: disabled.contains(p)) {
                        selection = p
                    }
                }
            }
            .padding(.horizontal, FFSpacing.lg)
            .padding(.vertical, 6)
        }
    }

    @ViewBuilder
    private func pill(title: String, active: Bool, disabled: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Text(title)
                .font(.subheadline.weight(.medium))
                .foregroundStyle(disabled ? FFColor.textMuted.opacity(0.5)
                    : (active ? FFColor.accent : FFColor.textSecondary))
                .padding(.horizontal, 14)
                .padding(.vertical, 6)
                .background(active ? FFColor.accentDim : FFColor.bgPrimary)
                .clipShape(Capsule())
                .overlay(Capsule().strokeBorder(active ? FFColor.accent : FFColor.border))
        }
        .buttonStyle(.plain)
        .disabled(disabled)
    }
}
