import SwiftUI

/// Per-raw-stat actual vs the four models (honors the model-display filter).
struct BreakdownTable: View {
    let breakdown: Breakdown
    let selectedModel: PredictionModel?

    private var models: [PredictionModel] {
        selectedModel.map { [$0] } ?? PredictionModel.allCases
    }

    var body: some View {
        if breakdown.unavailable || breakdown.components.isEmpty {
            Text("Per-stat breakdown unavailable for this week.")
                .font(.caption)
                .foregroundStyle(FFColor.textSecondary)
        } else {
            Grid(alignment: .leading, horizontalSpacing: FFSpacing.md, verticalSpacing: 6) {
                GridRow {
                    Text("Stat").font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                    Text("Actual").font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                        .gridColumnAlignment(.trailing)
                    ForEach(models) { m in
                        Text(m.shortLabel).font(.caption2.weight(.semibold)).foregroundStyle(m.color)
                            .gridColumnAlignment(.trailing)
                    }
                }
                Divider().overlay(FFColor.border).gridCellColumns(2 + models.count)
                ForEach(breakdown.components) { component in
                    GridRow {
                        Text(component.label).font(.caption).foregroundStyle(FFColor.textPrimary)
                        Text(cell(component.actual, component.unit))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(FFColor.textPrimary)
                            .gridColumnAlignment(.trailing)
                        ForEach(models) { m in
                            Text(cell(component.value(for: m), component.unit))
                                .font(.caption.monospacedDigit())
                                .foregroundStyle(FFColor.textSecondary)
                                .gridColumnAlignment(.trailing)
                        }
                    }
                }
            }
        }
    }

    private func cell(_ value: Double?, _ unit: String) -> String {
        guard value != nil else { return "--" }
        return unit.isEmpty ? Fmt.num(value, 1) : "\(Fmt.num(value, 1)) \(unit)"
    }
}
