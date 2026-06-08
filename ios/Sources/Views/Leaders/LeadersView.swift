import SwiftUI

/// Season Leaders — snapshot-hydrated, client-filtered/sorted card list.
struct LeadersView: View {
    @Environment(AppState.self) private var app
    @State private var store = SnapshotStore()
    @State private var leaders = LeadersStore()
    @State private var search = ""

    var body: some View {
        content
            .navigationTitle("Leaders")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    ModelDisplayMenu()
                    ScoringMenu()
                }
            }
            .searchable(text: $search, prompt: "Search players")
            .task { if !store.hasData { await store.hydrate(scoring: app.scoring) } }
            .onChange(of: app.scoring) {
                if !store.usingSnapshot { Task { await store.ensureLive(app.scoring) } }
            }
            .refreshable { await store.hydrate(scoring: app.scoring) }
    }

    @ViewBuilder
    private var content: some View {
        if store.isLoading, !store.hasData {
            SkeletonList()
        } else if let error = store.errorMessage, !store.hasData {
            EmptyStateView(icon: "wifi.exclamationmark", title: "Couldn't load", message: error,
                           retry: { Task { await store.hydrate(scoring: app.scoring) } })
        } else {
            list
        }
    }

    private var list: some View {
        @Bindable var leaders = leaders
        let rows = leaders.filteredSorted(store.players(app.scoring), search: search)
        return List {
            if !store.degradedPositions.isEmpty {
                DegradedBanner(positions: store.degradedPositions)
                    .listRowInsets(EdgeInsets(top: 4, leading: FFSpacing.lg, bottom: 4, trailing: FFSpacing.lg))
                    .listRowBackground(Color.clear)
            }
            Text("\(rows.count) player-week\(rows.count == 1 ? "" : "s")")
                .font(.caption).foregroundStyle(FFColor.textMuted)
                .listRowBackground(Color.clear)
            ForEach(rows) { player in
                NavigationLink(value: PlayerRoute(playerID: player.playerID, name: player.name, week: player.week)) {
                    LeaderRow(player: player, selectedModel: app.selectedModel)
                }
                .listRowBackground(FFColor.bgSecondary)
            }
            if rows.isEmpty {
                Text("No players match your filters.")
                    .font(.subheadline).foregroundStyle(FFColor.textSecondary)
                    .listRowBackground(Color.clear)
            }
        }
        .listStyle(.plain)
        .ffScreenBackground()
        .safeAreaInset(edge: .top) { filterBar }
    }

    private var filterBar: some View {
        @Bindable var leaders = leaders
        return VStack(spacing: FFSpacing.sm) {
            PositionPills(selection: $leaders.position)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: FFSpacing.sm) {
                    Menu {
                        Picker("Sort by", selection: $leaders.sortKey) {
                            ForEach(LeadersStore.SortKey.allCases) { Text($0.label).tag($0) }
                        }
                        Picker("Order", selection: $leaders.ascending) {
                            Text("High → Low").tag(false)
                            Text("Low → High").tag(true)
                        }
                    } label: { chip("Sort: \(leaders.sortKey.label)") }

                    Menu {
                        Picker("Week", selection: $leaders.week) {
                            Text("All weeks").tag(Int?.none)
                            ForEach(store.weeks, id: \.self) { Text("Week \($0)").tag(Int?.some($0)) }
                        }
                    } label: { chip(leaders.week.map { "Wk \($0)" } ?? "Week") }

                    Menu {
                        Picker("Team", selection: $leaders.team) {
                            Text("All teams").tag(String?.none)
                            ForEach(store.teams, id: \.self) { Text($0).tag(String?.some($0)) }
                        }
                    } label: { chip(leaders.team ?? "Team") }

                    Menu {
                        Picker("Min projected", selection: $leaders.minPoints) {
                            Text("Any").tag(Double?.none)
                            ForEach([5.0, 10.0, 15.0, 20.0], id: \.self) { Text("≥ \(Int($0)) pts").tag(Double?.some($0)) }
                        }
                    } label: { chip(leaders.minPoints.map { "≥ \(Int($0))" } ?? "Min pts") }
                }
                .padding(.horizontal, FFSpacing.lg)
            }
        }
        .padding(.bottom, 6)
        .background(FFColor.bgPrimary)
        .overlay(alignment: .bottom) { Divider().overlay(FFColor.border) }
    }

    private func chip(_ text: String) -> some View {
        HStack(spacing: 4) {
            Text(text)
            Image(systemName: "chevron.down").font(.caption2)
        }
        .font(.caption.weight(.medium))
        .foregroundStyle(FFColor.textSecondary)
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(FFColor.bgSecondary)
        .clipShape(Capsule())
        .overlay(Capsule().strokeBorder(FFColor.border))
    }
}
