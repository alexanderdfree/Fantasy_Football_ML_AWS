import SwiftUI

/// Yellow notice shown when some positions failed to load (`degraded_positions`).
struct DegradedBanner: View {
    let positions: [String]

    var body: some View {
        if !positions.isEmpty {
            HStack(alignment: .top, spacing: FFSpacing.sm) {
                Image(systemName: "exclamationmark.triangle.fill")
                    .foregroundStyle(FFColor.yellow)
                Text("Heads up: predictions unavailable for \(positions.joined(separator: ", ")). Showing last updated data for the other positions.")
                    .font(.caption)
                    .foregroundStyle(FFColor.textSecondary)
            }
            .padding(FFSpacing.md)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(FFColor.yellowDim)
            .clipShape(RoundedRectangle(cornerRadius: FFRadius.sm, style: .continuous))
        }
    }
}
