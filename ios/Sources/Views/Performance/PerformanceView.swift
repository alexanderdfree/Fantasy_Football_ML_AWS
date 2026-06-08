import SwiftUI

/// Model Performance — overall metric cards, MAE/R²-by-position bars, weekly MAE
/// line, and a per-position target breakdown.
struct PerformanceView: View {
    @Environment(AppState.self) private var app
    @State private var store = MetricsStore()
    @State private var perfPosition: Position = .qb

    var body: some View {
        ScrollView {
            VStack(spacing: FFSpacing.lg) {
                LoadStateView(state: store.metrics, retry: { Task { await store.load(scoring: app.scoring) } }) { metrics in
                    VStack(spacing: FFSpacing.lg) {
                        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: FFSpacing.md) {
                            ForEach(PredictionModel.allCases) { model in
                                MetricCard(model: model, overall: metrics.metrics(for: model)?.overall)
                            }
                        }
                        Card { PositionMetricBarChart(metrics: metrics, metric: .mae, title: "MAE by Position", caption: "Lower is better") }
                        Card { PositionMetricBarChart(metrics: metrics, metric: .r2, title: "R² by Position", caption: "Higher is better") }
                        ModelLegend()
                    }
                }

                if let weekly = store.weekly.value {
                    Card { WeeklyMaeLineChart(weekly: weekly) }
                }

                if let details = store.positionDetails.value {
                    PositionBreakdownCard(
                        details: details,
                        position: $perfPosition,
                        scoring: app.scoring,
                        selectedModel: app.selectedModel
                    )
                }
            }
            .padding(FFSpacing.lg)
        }
        .background(FFColor.bgPrimary)
        .navigationTitle("Accuracy")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                ModelDisplayMenu()
                ScoringMenu()
            }
        }
        .task { if store.metrics.value == nil { await store.load(scoring: app.scoring) } }
        .onChange(of: app.scoring) { Task { await store.load(scoring: app.scoring) } }
        .refreshable { await store.load(scoring: app.scoring) }
    }
}
