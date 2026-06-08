import SwiftUI

/// One 80% prediction-interval example: floor–ceiling fill, median tick, and the
/// actual outcome marker (green if in-band, red if out). Ports `renderBandBar`.
struct BandBar: View {
    let example: Comparison.BandExample

    var body: some View {
        let lo = min(example.floor, example.actual)
        let hi = max(example.ceiling, example.actual)
        let span = max(hi - lo, 0.0001)
        func pct(_ x: Double) -> Double { (x - lo) / span }

        return VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(example.playerName).font(.caption.weight(.semibold)).foregroundStyle(FFColor.textPrimary).lineLimit(1)
                Spacer()
                Text("Wk \(example.week)").font(.caption2).foregroundStyle(FFColor.textMuted)
                Text("proj \(Fmt.num(example.projection, 1))").font(.caption2).foregroundStyle(FFColor.textSecondary)
            }
            GeometryReader { geo in
                let w = geo.size.width
                ZStack(alignment: .leading) {
                    Capsule().fill(FFColor.bgPrimary).frame(height: 6)
                    Capsule().fill(FFColor.accentDim)
                        .frame(width: max(w * (pct(example.ceiling) - pct(example.floor)), 1), height: 6)
                        .offset(x: w * pct(example.floor))
                    Rectangle().fill(FFColor.textSecondary).frame(width: 2, height: 12)
                        .offset(x: w * pct(example.median) - 1)
                    Circle().fill(example.inBand ? FFColor.accent : FFColor.red)
                        .frame(width: 9, height: 9)
                        .offset(x: w * pct(example.actual) - 4.5)
                }
            }
            .frame(height: 12)
            HStack {
                Text(Fmt.num(example.floor, 1)).font(.caption2).foregroundStyle(FFColor.textMuted)
                Spacer()
                Text("actual \(Fmt.num(example.actual, 1)) \(example.inBand ? "✓" : "✗")")
                    .font(.caption2)
                    .foregroundStyle(example.inBand ? FFColor.accent : FFColor.red)
                Spacer()
                Text(Fmt.num(example.ceiling, 1)).font(.caption2).foregroundStyle(FFColor.textMuted)
            }
        }
        .padding(.vertical, 4)
    }
}
