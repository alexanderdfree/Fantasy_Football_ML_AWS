import SwiftUI

/// A colored dot + short label identifying a model (matches the chart series).
struct ModelChip: View {
    let model: PredictionModel
    var showLabel: Bool = true

    var body: some View {
        HStack(spacing: 4) {
            Circle().fill(model.color).frame(width: 8, height: 8)
            if showLabel {
                Text(model.shortLabel)
                    .font(.caption2)
                    .foregroundStyle(FFColor.textSecondary)
            }
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel(model.metricsDisplayName)
    }
}

/// Horizontal legend of the four models (+ Actual where relevant).
struct ModelLegend: View {
    var includeActual: Bool = false
    var models: [PredictionModel] = PredictionModel.allCases

    var body: some View {
        HStack(spacing: FFSpacing.md) {
            if includeActual {
                HStack(spacing: 4) {
                    Circle().fill(FFColor.seriesActual).frame(width: 8, height: 8)
                    Text("Actual").font(.caption2).foregroundStyle(FFColor.textSecondary)
                }
            }
            ForEach(models) { ModelChip(model: $0) }
        }
    }
}
