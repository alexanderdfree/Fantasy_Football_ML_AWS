import SwiftUI

/// Pushed player screen — headshot/header, season avg/total, weekly Actual-vs-
/// predicted chart, and (when a week is known) the per-stat breakdown.
struct PlayerDetailView: View {
    let route: PlayerRoute
    @Environment(AppState.self) private var app
    @State private var store = PlayerDetailStore()

    var body: some View {
        ScrollView {
            LoadStateView(state: store.detail, retry: { Task { await reload() } }) { detail in
                VStack(spacing: FFSpacing.lg) {
                    header(detail)
                    HStack(spacing: FFSpacing.md) {
                        StatCard(title: "Season Avg", value: Fmt.num(detail.seasonAvg))
                        StatCard(title: "Season Total", value: Fmt.num(detail.seasonTotal))
                    }
                    Card {
                        VStack(alignment: .leading, spacing: FFSpacing.md) {
                            SectionLabel(text: "Weekly Fantasy Points")
                            PlayerWeeklyChart(weekly: detail.weekly)
                            ModelLegend(includeActual: true)
                        }
                    }
                    breakdownCard
                }
                .padding(FFSpacing.lg)
            }
        }
        .background(FFColor.bgPrimary)
        .navigationTitle(route.name ?? "Player")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar { ToolbarItem(placement: .topBarTrailing) { ScoringMenu() } }
        .task { await reload() }
        .onChange(of: app.scoring) { Task { await reload() } }
    }

    private func header(_ detail: PlayerDetail) -> some View {
        HStack(spacing: FFSpacing.md) {
            Headshot(url: detail.headshot, name: detail.name, position: detail.position, size: 64)
            VStack(alignment: .leading, spacing: 4) {
                Text(detail.name).font(.title2.weight(.bold)).foregroundStyle(FFColor.textPrimary)
                HStack(spacing: 6) {
                    PositionBadge(position: detail.position, size: .caption)
                    Text(detail.team).font(.subheadline).foregroundStyle(FFColor.textSecondary)
                }
            }
            Spacer()
        }
    }

    @ViewBuilder
    private var breakdownCard: some View {
        if let week = route.week, let state = store.breakdown {
            Card {
                VStack(alignment: .leading, spacing: FFSpacing.md) {
                    SectionLabel(text: "Week \(week) Breakdown")
                    switch state {
                    case .idle, .loading:
                        ProgressView().tint(FFColor.accent).frame(maxWidth: .infinity)
                    case let .loaded(breakdown):
                        BreakdownTable(breakdown: breakdown, selectedModel: app.selectedModel)
                    case let .failed(message):
                        Text(message).font(.caption).foregroundStyle(FFColor.textSecondary)
                    }
                }
            }
        }
    }

    private func reload() async {
        await store.load(playerID: route.playerID, week: route.week, scoring: app.scoring)
    }
}
