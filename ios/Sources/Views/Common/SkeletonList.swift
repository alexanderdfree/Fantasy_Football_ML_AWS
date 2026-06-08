import SwiftUI

/// Redacted placeholder rows shown during first load (lighter than a spinner).
struct SkeletonList: View {
    var rows: Int = 8

    var body: some View {
        VStack(spacing: 10) {
            ForEach(0..<rows, id: \.self) { _ in
                HStack(spacing: FFSpacing.md) {
                    Circle().fill(FFColor.bgCard).frame(width: 36, height: 36)
                    VStack(alignment: .leading, spacing: 6) {
                        RoundedRectangle(cornerRadius: 4).fill(FFColor.bgCard).frame(width: 150, height: 12)
                        RoundedRectangle(cornerRadius: 4).fill(FFColor.bgCard).frame(width: 90, height: 10)
                    }
                    Spacer()
                    RoundedRectangle(cornerRadius: 4).fill(FFColor.bgCard).frame(width: 44, height: 20)
                }
                .padding(.horizontal, FFSpacing.lg)
            }
        }
        .padding(.vertical, FFSpacing.sm)
        .redacted(reason: .placeholder)
        .accessibilityHidden(true)
    }
}
