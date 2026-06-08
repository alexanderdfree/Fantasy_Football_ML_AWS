import Charts
import SwiftUI

/// Week-by-week Actual (solid) vs the four model predictions (dashed).
struct PlayerWeeklyChart: View {
    let weekly: [PlayerDetail.WeeklyPoint]

    private struct CP: Identifiable {
        let week: Int
        let value: Double
        var id: Int { week }
    }

    private var actual: [CP] {
        weekly.compactMap { p in p.actual.map { CP(week: p.week, value: $0) } }
    }

    private func series(_ model: PredictionModel) -> [CP] {
        weekly.compactMap { p in p.prediction(for: model).map { CP(week: p.week, value: $0) } }
    }

    var body: some View {
        Chart {
            ForEach(actual) { pt in
                LineMark(x: .value("Week", pt.week), y: .value("FP", pt.value))
                    .foregroundStyle(by: .value("Series", "Actual"))
                    .lineStyle(StrokeStyle(lineWidth: 2.5))
                    .interpolationMethod(.catmullRom)
                PointMark(x: .value("Week", pt.week), y: .value("FP", pt.value))
                    .foregroundStyle(by: .value("Series", "Actual"))
                    .symbolSize(45)
            }
            ForEach(PredictionModel.allCases) { model in
                ForEach(series(model)) { pt in
                    LineMark(x: .value("Week", pt.week), y: .value("FP", pt.value))
                        .foregroundStyle(by: .value("Series", model.shortLabel))
                        .lineStyle(StrokeStyle(lineWidth: 2, dash: [6, 3]))
                        .interpolationMethod(.catmullRom)
                }
            }
        }
        .chartForegroundStyleScale([
            "Actual": FFColor.seriesActual,
            "Ridge": FFColor.modelRidge,
            "NN": FFColor.modelNN,
            "Attn NN": FFColor.modelAttnNN,
            "LGBM": FFColor.modelLGBM,
        ])
        .chartLegend(.hidden)
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 6)) { _ in
                AxisGridLine().foregroundStyle(FFColor.border)
                AxisValueLabel().foregroundStyle(FFColor.textSecondary)
            }
        }
        .chartYAxis {
            AxisMarks { _ in
                AxisGridLine().foregroundStyle(FFColor.border)
                AxisValueLabel().foregroundStyle(FFColor.textSecondary)
            }
        }
        .frame(height: 240)
    }
}
