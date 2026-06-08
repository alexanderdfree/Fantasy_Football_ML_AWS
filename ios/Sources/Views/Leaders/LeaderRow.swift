import SwiftUI

/// Card row for Season Leaders: player + week + actual, then the four models
/// (or one, per the model-display filter) with signed error vs actual.
struct LeaderRow: View {
    let player: Player
    let selectedModel: PredictionModel?

    var body: some View {
        VStack(spacing: FFSpacing.sm) {
            HStack(spacing: FFSpacing.sm) {
                Headshot(url: player.headshot, name: player.name, position: player.position, size: 32)
                VStack(alignment: .leading, spacing: 2) {
                    Text(player.name).font(.subheadline.weight(.semibold)).foregroundStyle(FFColor.textPrimary).lineLimit(1)
                    HStack(spacing: 6) {
                        PositionBadge(position: player.position)
                        Text(player.team).font(.caption2).foregroundStyle(FFColor.textSecondary)
                    }
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Wk \(player.week)").font(.caption2).foregroundStyle(FFColor.textMuted)
                    HStack(spacing: 4) {
                        Text("ACT").font(.caption2).foregroundStyle(FFColor.textMuted)
                        Text(Fmt.num(player.actual)).font(.headline).monospacedDigit().foregroundStyle(FFColor.textPrimary)
                    }
                }
            }
            modelsRow
        }
        .padding(.vertical, 2)
    }

    @ViewBuilder
    private var modelsRow: some View {
        if let model = selectedModel {
            HStack {
                ModelChip(model: model)
                Spacer()
                Text(Fmt.num(player.prediction(for: model))).font(.subheadline).monospacedDigit().foregroundStyle(FFColor.textPrimary)
                ErrorDeltaText(pred: player.prediction(for: model), actual: player.actual)
            }
        } else {
            HStack(spacing: 0) {
                ForEach(PredictionModel.allCases) { model in
                    VStack(spacing: 2) {
                        ModelChip(model: model)
                        Text(Fmt.num(player.prediction(for: model))).font(.caption).monospacedDigit().foregroundStyle(FFColor.textPrimary)
                        ErrorDeltaText(pred: player.prediction(for: model), actual: player.actual, font: .caption2)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }
}
