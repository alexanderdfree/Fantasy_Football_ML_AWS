import SwiftUI

/// Signed prediction error (pred − actual): muted when |Δ| < 1, else green for
/// over-prediction / red for under (ports the web `deltaClass`).
struct ErrorDeltaText: View {
    let pred: Double?
    let actual: Double?
    var font: Font = .caption

    var body: some View {
        if let d = Fmt.errDelta(pred, actual) {
            Text(Fmt.delta(d))
                .font(font.monospacedDigit())
                .foregroundStyle(color(for: d))
                .accessibilityLabel("error \(Fmt.delta(d))")
        } else {
            Text("--")
                .font(font.monospacedDigit())
                .foregroundStyle(FFColor.textMuted)
        }
    }

    private func color(for d: Double) -> Color {
        if abs(d) < 1 { return FFColor.textMuted }
        return d > 0 ? FFColor.accent : FFColor.red
    }
}
