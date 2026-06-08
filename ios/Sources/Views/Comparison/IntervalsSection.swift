import SwiftUI

/// 80% prediction-interval calibration grid + example bands per position.
struct IntervalsSection: View {
    let intervals: Comparison.Intervals
    @State private var position: Position = .qb

    private let experts: [(key: String, label: String)] = [("nflcom", "NFL.com"), ("rotowire", "RotoWire")]

    var body: some View {
        VStack(alignment: .leading, spacing: FFSpacing.md) {
            Text("Does the 80% band contain about 80% of outcomes? Near 80% is well-calibrated.")
                .font(.caption).foregroundStyle(FFColor.textSecondary)

            Grid(alignment: .leading, horizontalSpacing: FFSpacing.md, verticalSpacing: 6) {
                GridRow {
                    Text("Pos").font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                    ForEach(experts, id: \.key) { e in
                        Text(e.label).font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                            .gridColumnAlignment(.trailing)
                    }
                }
                ForEach(Position.displayOrder) { p in
                    GridRow {
                        Text(p.rawValue).font(.caption).foregroundStyle(FFColor.textPrimary)
                        ForEach(experts, id: \.key) { e in
                            calibrationCell(intervals.block(source: e.key, position: p.rawValue))
                                .gridColumnAlignment(.trailing)
                        }
                    }
                }
            }

            Divider().overlay(FFColor.border)

            Picker("Position", selection: $position) {
                ForEach(Position.displayOrder) { Text($0.rawValue).tag($0) }
            }
            .pickerStyle(.segmented)

            ForEach(experts, id: \.key) { e in
                if let block = intervals.block(source: e.key, position: position.rawValue),
                   let examples = block.examples, !examples.isEmpty {
                    Text(e.label).font(.caption.weight(.semibold)).foregroundStyle(FFColor.textPrimary)
                    ForEach(examples.prefix(3)) { BandBar(example: $0) }
                }
            }
        }
    }

    @ViewBuilder
    private func calibrationCell(_ block: Comparison.IntervalBlock?) -> some View {
        if let cal = block?.calibration {
            VStack(alignment: .trailing, spacing: 1) {
                Text("\(Int((cal.coverage * 100).rounded()))%")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(cal.flag == "ok" ? FFColor.accent : FFColor.yellow)
                if let n = cal.nEval {
                    Text("n=\(n)").font(.caption2).foregroundStyle(FFColor.textMuted)
                }
            }
        } else {
            Text("—").font(.caption).foregroundStyle(FFColor.textMuted)
        }
    }
}
