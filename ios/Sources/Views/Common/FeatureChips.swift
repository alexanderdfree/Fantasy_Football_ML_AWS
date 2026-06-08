import SwiftUI

/// Wrapping chip row for feature names (Performance + Architecture).
struct FeatureChips: View {
    let features: [String]

    var body: some View {
        FlowLayout(spacing: 6) {
            ForEach(features, id: \.self) { feature in
                Text(feature)
                    .font(.caption2)
                    .foregroundStyle(FFColor.textSecondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(FFColor.bgPrimary)
                    .clipShape(Capsule())
                    .overlay(Capsule().strokeBorder(FFColor.border))
            }
        }
    }
}
