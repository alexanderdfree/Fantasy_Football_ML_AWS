import Charts
import SwiftUI

/// Grouped bars: one cluster per position, one bar per model (MAE or R²).
struct PositionMetricBarChart: View {
    let metrics: MetricsResponse
    let metric: MetricKind
    let title: String
    let caption: String

    private struct Bar: Identifiable {
        let position: String
        let model: String
        let value: Double
        var id: String { "\(position)-\(model)" }
    }

    private var bars: [Bar] {
        var out: [Bar] = []
        for pos in Position.displayOrder {
            for model in PredictionModel.allCases {
                if let row = metrics.metrics(for: model)?.byPosition.first(where: { $0.position == pos.rawValue }),
                   let value = row.value(metric) {
                    out.append(Bar(position: pos.rawValue, model: model.shortLabel, value: value))
                }
            }
        }
        return out
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.subheadline.weight(.semibold)).foregroundStyle(FFColor.textPrimary)
            Text(caption).font(.caption2).foregroundStyle(FFColor.textMuted)
            Chart(bars) { bar in
                BarMark(
                    x: .value("Position", bar.position),
                    y: .value(metric.label, bar.value)
                )
                .foregroundStyle(by: .value("Model", bar.model))
                .position(by: .value("Model", bar.model))
            }
            .chartForegroundStyleScale(ChartStyle.modelColorScale)
            .chartLegend(.hidden)
            .chartXAxis {
                AxisMarks { _ in AxisValueLabel().foregroundStyle(FFColor.textSecondary) }
            }
            .chartYAxis {
                AxisMarks { _ in
                    AxisGridLine().foregroundStyle(FFColor.border)
                    AxisValueLabel().foregroundStyle(FFColor.textSecondary)
                }
            }
            .frame(height: 200)
        }
    }
}

/// Weekly MAE across the test season, one line per model.
struct WeeklyMaeLineChart: View {
    let weekly: WeeklyAccuracy

    private struct P: Identifiable {
        let week: Int
        let mae: Double
        let model: String
        var id: String { "\(model)-\(week)" }
    }

    private var points: [P] {
        var out: [P] = []
        for model in PredictionModel.allCases {
            let series = weekly.series(for: model)
            for (index, week) in weekly.weeks.enumerated() where index < series.count {
                if let value = series[index] {
                    out.append(P(week: week, mae: value, model: model.shortLabel))
                }
            }
        }
        return out
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Weekly MAE Across Test Season").font(.subheadline.weight(.semibold)).foregroundStyle(FFColor.textPrimary)
            Text("Lower is better").font(.caption2).foregroundStyle(FFColor.textMuted)
            Chart(points) { p in
                LineMark(x: .value("Week", p.week), y: .value("MAE", p.mae))
                    .foregroundStyle(by: .value("Model", p.model))
                    .interpolationMethod(.catmullRom)
            }
            .chartForegroundStyleScale(ChartStyle.modelColorScale)
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
            .frame(height: 200)
        }
    }
}
