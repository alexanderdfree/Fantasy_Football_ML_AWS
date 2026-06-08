import SwiftUI

/// The hero tab: next unplayed week's projected slate (skill positions only).
struct UpcomingView: View {
    @Environment(AppState.self) private var app
    @State private var store = UpcomingStore()
    @State private var position: Position? = nil
    @State private var search = ""

    var body: some View {
        content
            .navigationTitle("Next Week")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .principal) {
                    if case let .ready(week) = store.state, let label = week.weekLabel {
                        Text(label).font(.headline).foregroundStyle(FFColor.textPrimary)
                    }
                }
                ToolbarItemGroup(placement: .topBarTrailing) {
                    ModelDisplayMenu()
                    ScoringMenu()
                }
            }
            .searchable(text: $search, placement: .navigationBarDrawer(displayMode: .automatic), prompt: "Search players")
            .task {
                if case .loading = store.state { await store.load() }
            }
            .refreshable { await store.load() }
    }

    @ViewBuilder
    private var content: some View {
        switch store.state {
        case .loading:
            SkeletonList()
        case .warming:
            EmptyStateView(
                icon: "hourglass",
                title: "Building this week's projections…",
                message: "Check back in a minute.",
                retry: { Task { await store.load() } }
            )
        case let .offseason(message):
            EmptyStateView(
                icon: "calendar",
                title: "No upcoming games",
                message: message,
                retry: { Task { await store.load() } }
            )
        case let .failed(message):
            EmptyStateView(
                icon: "wifi.exclamationmark",
                title: "Couldn't load projections",
                message: message,
                retry: { Task { await store.load() } }
            )
        case let .ready(week):
            readyList(week)
        }
    }

    private func readyList(_ week: UpcomingWeek) -> some View {
        let rows = filteredSorted(week.players(app.scoring))
        return List {
            if let degraded = week.degradedPositions, !degraded.isEmpty {
                DegradedBanner(positions: degraded)
                    .listRowInsets(EdgeInsets(top: 4, leading: FFSpacing.lg, bottom: 4, trailing: FFSpacing.lg))
                    .listRowBackground(Color.clear)
            }
            if let updated = Fmt.relativeTime(fromISO: week.generatedAt) {
                Text("Updated \(updated) · \(rows.count) player\(rows.count == 1 ? "" : "s")")
                    .font(.caption)
                    .foregroundStyle(FFColor.textMuted)
                    .listRowBackground(Color.clear)
            }
            ForEach(rows) { player in
                NavigationLink(value: PlayerRoute(playerID: player.playerID, name: player.name)) {
                    UpcomingPlayerCard(player: player, scoring: app.scoring, selectedModel: app.selectedModel)
                }
                .listRowBackground(FFColor.bgSecondary)
            }
            if rows.isEmpty {
                Text("No players match your filters.")
                    .font(.subheadline)
                    .foregroundStyle(FFColor.textSecondary)
                    .listRowBackground(Color.clear)
            }
        }
        .listStyle(.plain)
        .ffScreenBackground()
        .safeAreaInset(edge: .top) {
            VStack(spacing: 0) {
                PositionPills(selection: $position, disabled: [.k, .dst])
                Divider().overlay(FFColor.border)
            }
            .background(FFColor.bgPrimary)
        }
    }

    private func filteredSorted(_ players: [UpcomingPlayer]) -> [UpcomingPlayer] {
        players
            .filter { p in
                (position == nil || p.position == position?.rawValue)
                    && (search.isEmpty || p.name.localizedCaseInsensitiveContains(search))
            }
            .sorted { a, b in
                switch (a.bestProjection, b.bestProjection) {
                case let (x?, y?): return x > y
                case (nil, _): return false
                case (_, nil): return true
                }
            }
    }
}
