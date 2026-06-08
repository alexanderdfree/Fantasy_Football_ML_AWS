import SwiftUI

/// Overall MAE / RMSE / R² tile for one model.
struct MetricCard: View {
    let model: PredictionModel
    let overall: MetricBlock?

    var body: some View {
        VStack(alignment: .leading, spacing: FFSpacing.sm) {
            HStack(spacing: 6) {
                Circle().fill(model.color).frame(width: 8, height: 8)
                Text(model.metricsDisplayName)
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(FFColor.textPrimary)
                    .lineLimit(1)
                    .minimumScaleFactor(0.8)
            }
            row("MAE", overall?.mae)
            row("RMSE", overall?.rmse)
            row("R²", overall?.r2)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(FFSpacing.md)
        .background(FFColor.bgCard)
        .clipShape(RoundedRectangle(cornerRadius: FFRadius.lg, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: FFRadius.lg, style: .continuous).strokeBorder(FFColor.border))
    }

    private func row(_ label: String, _ value: Double?) -> some View {
        HStack {
            Text(label).font(.caption).foregroundStyle(FFColor.textMuted)
            Spacer()
            Text(Fmt.num(value, 3)).font(.callout).monospacedDigit().foregroundStyle(FFColor.textPrimary)
        }
    }
}
