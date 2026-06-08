import SwiftUI

/// Marquee card for the Upcoming-Week slate: player + matchup, Vegas line, and
/// the four projections (no actual / error — games haven't been played).
struct UpcomingPlayerCard: View {
    let player: UpcomingPlayer
    let scoring: ScoringFormat
    let selectedModel: PredictionModel?

    var body: some View {
        VStack(alignment: .leading, spacing: FFSpacing.md) {
            HStack(spacing: FFSpacing.sm) {
                Headshot(url: player.headshot, name: player.name, position: player.position, size: 40)
                VStack(alignment: .leading, spacing: 2) {
                    Text(player.name).font(.headline).foregroundStyle(FFColor.textPrimary).lineLimit(1)
                    HStack(spacing: 6) {
                        PositionBadge(position: player.position)
                        Text(player.team).font(.caption).foregroundStyle(FFColor.textSecondary)
                    }
                }
                Spacer()
                Text(player.matchupLabel)
                    .font(.subheadline.weight(.medium))
                    .foregroundStyle(FFColor.textSecondary)
            }

            VegasLineView(
                spread: player.spreadLine,
                total: player.totalLine,
                impliedTeamTotal: player.impliedTeamTotal
            )

            projections
        }
        .padding(.vertical, FFSpacing.xs)
    }

    @ViewBuilder
    private var projections: some View {
        if let model = selectedModel {
            HStack {
                ModelChip(model: model)
                Spacer()
                Text(Fmt.num(player.prediction(for: model)))
                    .font(.title3.weight(.bold))
                    .monospacedDigit()
                    .foregroundStyle(FFColor.textPrimary)
            }
        } else {
            HStack(spacing: 0) {
                ForEach(PredictionModel.allCases) { model in
                    VStack(spacing: 3) {
                        ModelChip(model: model)
                        Text(Fmt.num(player.prediction(for: model)))
                            .font(.subheadline.weight(.semibold))
                            .monospacedDigit()
                            .foregroundStyle(FFColor.textPrimary)
                    }
                    .frame(maxWidth: .infinity)
                }
            }
        }
    }
}
