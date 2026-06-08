import SwiftUI

/// Per-(position, model) improve/regress tint vs the most recent earlier run that
/// trained the same pair (ports annotateHistoryDeltas — walks oldest→newest with
/// a lastSeen map, since runs retrain only a subset of positions).
struct HistoryDeltas {
    enum Tint { case improve, regress }
    private var map: [String: Tint] = [:]

    init(rows: [BenchmarkHistory.Row], metric: MetricKind) {
        let eps = 0.005
        var lastSeen: [String: Double] = [:]
        for row in rows.reversed() {
            for model in PredictionModel.allCases {
                for pill in row.pills(for: model) {
                    guard let cur = pill.value(metric) else { continue }
                    let key = "\(pill.position)|\(model.rawValue)"
                    if let prev = lastSeen[key] {
                        let delta = cur - prev
                        if delta <= -eps { map["\(row.id)|\(key)"] = .improve }
                        else if delta >= eps { map["\(row.id)|\(key)"] = .regress }
                    }
                    lastSeen[key] = cur
                }
            }
        }
    }

    func tint(rowID: String, position: String, model: PredictionModel) -> Tint? {
        map["\(rowID)|\(position)|\(model.rawValue)"]
    }
}

struct HistoryPill: View {
    let label: String
    let value: Double?
    var tint: HistoryDeltas.Tint?
    var isBest: Bool = false

    var body: some View {
        HStack(spacing: 4) {
            Text(label).font(.caption2).foregroundStyle(FFColor.textMuted)
            Text(value.map { Fmt.num($0, 2) } ?? "--")
                .font(.caption2).monospacedDigit()
                .foregroundStyle(color)
                .fontWeight(isBest ? .bold : .regular)
        }
        .padding(.horizontal, 6).padding(.vertical, 3)
        .background(background)
        .clipShape(Capsule())
    }

    private var color: Color {
        switch tint {
        case .improve: return FFColor.accent
        case .regress: return FFColor.red
        case nil: return FFColor.textPrimary
        }
    }

    private var background: Color {
        switch tint {
        case .improve: return FFColor.accentDim
        case .regress: return FFColor.redDim
        case nil: return FFColor.bgPrimary
        }
    }
}

struct HistoryView: View {
    @Environment(AppState.self) private var app
    @State private var store = HistoryStore()
    @State private var groupByModel = false
    @State private var detailed = false
    @State private var metric: MetricKind = .mae

    var body: some View {
        LoadStateView(state: store.state, retry: { Task { await store.load() } }) { history in
            let visible = history.rows.filter { !$0.trainingSkipped }
            let deltas = HistoryDeltas(rows: history.rows, metric: metric)
            List {
                Section { controls }.listRowBackground(Color.clear)
                if visible.isEmpty {
                    Text("No benchmark runs yet.").font(.subheadline).foregroundStyle(FFColor.textSecondary)
                        .listRowBackground(Color.clear)
                }
                ForEach(visible) { row in
                    HistoryRunCard(
                        row: row, repoSlug: history.repoSlug, groupByModel: groupByModel,
                        metric: metric, detailed: detailed, deltas: deltas, history: history, scoring: app.scoring
                    )
                    .listRowBackground(Color.clear)
                    .listRowInsets(EdgeInsets(top: 4, leading: FFSpacing.lg, bottom: 4, trailing: FFSpacing.lg))
                }
            }
            .listStyle(.plain)
            .ffScreenBackground()
        }
        .navigationTitle("History")
        .navigationBarTitleDisplayMode(.inline)
        .task { if store.state.value == nil { await store.load() } }
        .refreshable { await store.load() }
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: FFSpacing.sm) {
            Picker("Metric", selection: $metric) {
                Text("MAE").tag(MetricKind.mae)
                Text("RMSE").tag(MetricKind.rmse)
            }
            .pickerStyle(.segmented)
            Toggle("Group by model", isOn: $groupByModel).font(.caption).tint(FFColor.accent)
            Toggle("Per-target detail", isOn: $detailed).font(.caption).tint(FFColor.accent)
            Text("Green = better than the prior run · red = worse"
                + (groupByModel ? "" : " · bold = best model"))
                .font(.caption2).foregroundStyle(FFColor.textMuted)
        }
    }
}

private struct ModelPill: Identifiable {
    let model: PredictionModel
    let pill: BenchmarkHistory.Pill
    var id: String { model.rawValue }
}

struct HistoryRunCard: View {
    let row: BenchmarkHistory.Row
    let repoSlug: String
    let groupByModel: Bool
    let metric: MetricKind
    let detailed: Bool
    let deltas: HistoryDeltas
    let history: BenchmarkHistory
    let scoring: ScoringFormat
    @State private var expanded = false

    private var hasDetail: Bool {
        PredictionModel.allCases.contains { row.pills(for: $0).contains { $0.perTarget != nil } }
    }

    var body: some View {
        Card {
            VStack(alignment: .leading, spacing: FFSpacing.sm) {
                header
                if groupByModel { groupedByModel } else { groupedByPosition }
                if detailed, hasDetail {
                    Button { withAnimation { expanded.toggle() } } label: {
                        Label(expanded ? "Hide per-target" : "Show per-target",
                              systemImage: expanded ? "chevron.up" : "chevron.down")
                            .font(.caption2)
                    }
                    .buttonStyle(.plain).foregroundStyle(FFColor.accent)
                    if expanded { detailBlocks }
                }
            }
        }
    }

    private var header: some View {
        HStack {
            if let pr = row.prNumber, !repoSlug.isEmpty,
               let url = URL(string: "https://github.com/\(repoSlug)/pull/\(pr)") {
                Link("#\(pr)", destination: url).font(.caption.weight(.semibold)).tint(FFColor.accent)
            } else if let pr = row.prNumber {
                Text("#\(pr)").font(.caption.weight(.semibold)).foregroundStyle(FFColor.textSecondary)
            } else if let hash = row.gitHash {
                Text(hash).font(.caption.monospaced()).foregroundStyle(FFColor.textSecondary)
            }
            Spacer()
            Text(Fmt.historyTimestamp(row.timestamp)).font(.caption2).foregroundStyle(FFColor.textMuted)
            Text(Fmt.trainingTime(row.totalElapsedSec)).font(.caption2).foregroundStyle(FFColor.textMuted)
        }
    }

    private var groupedByPosition: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(Position.displayOrder) { pos in
                let entries = PredictionModel.allCases.compactMap { model in
                    row.pills(for: model).first { $0.position == pos.rawValue }.map { ModelPill(model: model, pill: $0) }
                }
                if !entries.isEmpty {
                    let best = entries.compactMap { $0.pill.value(metric) }.min()
                    HStack(alignment: .top, spacing: 6) {
                        Text(pos.rawValue).font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                            .frame(width: 34, alignment: .leading)
                        FlowLayout(spacing: 4) {
                            ForEach(entries) { entry in
                                HistoryPill(
                                    label: entry.model.shortLabel,
                                    value: entry.pill.value(metric),
                                    tint: deltas.tint(rowID: row.id, position: pos.rawValue, model: entry.model),
                                    isBest: isBest(entry.pill.value(metric), best)
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private var groupedByModel: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(PredictionModel.allCases) { model in
                let pills = row.pills(for: model)
                if !pills.isEmpty {
                    HStack(alignment: .top, spacing: 6) {
                        Text(model.shortLabel).font(.caption2.weight(.semibold)).foregroundStyle(model.color)
                            .frame(width: 52, alignment: .leading)
                        FlowLayout(spacing: 4) {
                            ForEach(pills, id: \.position) { pill in
                                HistoryPill(
                                    label: pill.position,
                                    value: pill.value(metric),
                                    tint: deltas.tint(rowID: row.id, position: pill.position, model: model)
                                )
                            }
                        }
                    }
                }
            }
        }
    }

    private var detailBlocks: some View {
        VStack(alignment: .leading, spacing: FFSpacing.md) {
            ForEach(Position.displayOrder) { pos in
                if let targets = targetKeys(for: pos) {
                    VStack(alignment: .leading, spacing: 4) {
                        PositionBadge(position: pos.rawValue)
                        Grid(alignment: .leading, horizontalSpacing: FFSpacing.sm, verticalSpacing: 4) {
                            GridRow {
                                Text("Target").font(.caption2.weight(.semibold)).foregroundStyle(FFColor.textMuted)
                                ForEach(PredictionModel.allCases) { m in
                                    Text(m.shortLabel).font(.caption2).foregroundStyle(m.color).gridColumnAlignment(.trailing)
                                }
                            }
                            ForEach(targets, id: \.self) { tkey in
                                GridRow {
                                    Text(history.targetLabels[tkey] ?? tkey).font(.caption2).foregroundStyle(FFColor.textPrimary)
                                    ForEach(PredictionModel.allCases) { m in
                                        let value = row.pills(for: m).first { $0.position == pos.rawValue }?.perTargetMap(metric)?[tkey]
                                        Text(Fmt.targetMae(value, targetKey: tkey, unit: history.targetUnits[tkey], scoring: scoring))
                                            .font(.caption2).monospacedDigit().foregroundStyle(FFColor.textSecondary)
                                            .gridColumnAlignment(.trailing)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func targetKeys(for pos: Position) -> [String]? {
        for model in PredictionModel.allCases {
            if let pt = row.pills(for: model).first(where: { $0.position == pos.rawValue })?.perTarget {
                return pt.keys.sorted()
            }
        }
        return nil
    }

    private func isBest(_ value: Double?, _ best: Double?) -> Bool {
        guard let value, let best else { return false }
        return abs(value - best) < 1e-9
    }
}
