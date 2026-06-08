import SwiftUI

/// Per-position model detail: target-by-model MAE grid (with FP equivalents),
/// position-specific feature chips, and the NN architecture summary line.
struct PositionBreakdownCard: View {
    let details: PositionDetailsResponse
    @Binding var position: Position
    let scoring: ScoringFormat
    let selectedModel: PredictionModel?

    private var models: [PredictionModel] { selectedModel.map { [$0] } ?? PredictionModel.allCases }
    private var posBinding: Binding<Position?> {
        Binding(get: { position }, set: { if let new = $0 { position = new } })
    }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: FFSpacing.md) {
                SectionLabel(text: "Per-Position Breakdown")
                PositionPills(selection: posBinding, includeAll: false)
                if let detail = details[position.rawValue] {
                    content(detail)
                } else {
                    Text("No data for \(position.rawValue).")
                        .font(.caption).foregroundStyle(FFColor.textSecondary)
                }
            }
        }
    }

    @ViewBuilder
    private func content(_ detail: PositionDetail) -> some View {
        VStack(alignment: .leading, spacing: FFSpacing.md) {
            HStack {
                PositionBadge(position: position.rawValue)
                Text(detail.label).font(.subheadline.weight(.semibold)).foregroundStyle(FFColor.textPrimary)
                Spacer()
                if let nf = detail.nFeatures, let ns = detail.nSamplesTest {
                    Text("\(nf) feats · \(ns) test").font(.caption2).foregroundStyle(FFColor.textMuted)
                }
            }

            Grid(alignment: .leading, horizontalSpacing: FFSpacing.sm, verticalSpacing: 6) {
                GridRow {
                    Text("Target").font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                    ForEach(models) { m in
                        Text(m.shortLabel).font(.caption2.weight(.semibold)).foregroundStyle(m.color)
                            .gridColumnAlignment(.trailing)
                    }
                }
                Divider().overlay(FFColor.border).gridCellColumns(1 + models.count)
                ForEach(detail.targets) { target in
                    let row = detail.targetMetrics?[target.key]
                    GridRow {
                        Text(target.label).font(.caption).foregroundStyle(FFColor.textPrimary)
                        ForEach(models) { m in
                            Text(Fmt.targetMae(row?.mae(for: m), targetKey: target.key, unit: row?.unit, scoring: scoring))
                                .font(.caption).monospacedDigit().foregroundStyle(FFColor.textSecondary)
                                .gridColumnAlignment(.trailing)
                        }
                    }
                }
                if let total = detail.targetMetrics?["total"] {
                    Divider().overlay(FFColor.border).gridCellColumns(1 + models.count)
                    GridRow {
                        Text("Total (FP)").font(.caption.weight(.bold)).foregroundStyle(FFColor.textPrimary)
                        ForEach(models) { m in
                            Text(Fmt.num(total.mae(for: m), 2)).font(.caption.weight(.bold)).monospacedDigit()
                                .foregroundStyle(FFColor.textPrimary).gridColumnAlignment(.trailing)
                        }
                    }
                }
            }

            if !detail.specificFeatures.isEmpty {
                SectionLabel(text: "Position-specific features")
                FeatureChips(features: detail.specificFeatures)
            }

            Text(archLine(detail))
                .font(.caption2).foregroundStyle(FFColor.textMuted)
        }
    }

    private func archLine(_ detail: PositionDetail) -> String {
        let backbone = detail.architecture.backbone.map(String.init).joined(separator: " → ")
        let head = detail.architecture.headHidden.map(String.init) ?? "?"
        return "Backbone [\(backbone)] → \(detail.targets.count) heads (hidden: \(head))"
    }
}
