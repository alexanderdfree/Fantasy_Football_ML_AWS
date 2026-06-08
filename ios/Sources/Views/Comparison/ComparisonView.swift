import SwiftUI

/// Comparison sources: our four models, then the two expert projection sources.
struct CmpSource: Identifiable {
    let key: String
    let label: String
    let color: Color
    let isModel: Bool
    var id: String { key }
}

let cmpSources: [CmpSource] =
    PredictionModel.allCases.map { CmpSource(key: $0.bareKey, label: $0.shortLabel, color: $0.color, isModel: true) }
    + [
        CmpSource(key: "nflcom", label: "NFL.com", color: FFColor.textSecondary, isModel: false),
        CmpSource(key: "rotowire", label: "RotoWire", color: FFColor.textSecondary, isModel: false),
    ]

private func valueIsBest(_ value: Double?, _ best: Double?) -> Bool {
    guard let value, let best else { return false }
    return abs(value - best) < 1e-9
}

/// Our models vs experts (NFL.com, RotoWire). PPR-pinned; MAE/RMSE/R² toggle.
struct ComparisonView: View {
    @State private var store = ComparisonStore()
    @State private var metric: MetricKind = .mae

    var body: some View {
        LoadStateView(state: store.state, retry: { Task { await store.load() } }) { comparison in
            List {
                Section("All rostered players") {
                    ForEach(Position.displayOrder) { pos in
                        ComparisonPositionGroup(comparison: comparison, subset: "all", position: pos, metric: metric)
                    }
                }
                .listRowBackground(FFColor.bgSecondary)

                Section("Top 30 per position") {
                    ForEach(Position.displayOrder) { pos in
                        ComparisonPositionGroup(comparison: comparison, subset: "top30", position: pos, metric: metric)
                    }
                }
                .listRowBackground(FFColor.bgSecondary)

                if comparison.expertReliability != nil {
                    Section("Reliability — residual σ (2025)") {
                        ForEach(Position.displayOrder) { pos in
                            ReliabilityGroup(comparison: comparison, position: pos)
                        }
                    }
                    .listRowBackground(FFColor.bgSecondary)
                }

                if let intervals = comparison.intervals {
                    Section("80% prediction intervals") {
                        IntervalsSection(intervals: intervals)
                    }
                    .listRowBackground(FFColor.bgSecondary)
                }

                Section("About this comparison") {
                    aboutNotes(comparison)
                }
                .listRowBackground(FFColor.bgSecondary)
            }
            .listStyle(.insetGrouped)
            .ffScreenBackground()
            .safeAreaInset(edge: .top) { metricBar }
        }
        .navigationTitle("Compare")
        .navigationBarTitleDisplayMode(.inline)
        .task { await store.load() }
    }

    private var metricBar: some View {
        VStack(spacing: 4) {
            Picker("Metric", selection: $metric) {
                ForEach([MetricKind.mae, .rmse, .r2]) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, FFSpacing.lg)
            Text(metric.hint).font(.caption2).foregroundStyle(FFColor.textMuted)
        }
        .padding(.vertical, FFSpacing.sm)
        .background(FFColor.bgPrimary)
        .overlay(alignment: .bottom) { Divider().overlay(FFColor.border) }
    }

    @ViewBuilder
    private func aboutNotes(_ comparison: Comparison) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            note("Seasons", "Trained 2012–2023, validated 2024, tested 2025. Every number here is on the held-out 2025 season; experts are scored on 2025 too.")
            note("Scoring", "Full PPR. Projections and actuals run through the same formula — apples-to-apples.")
            if let n = comparison.expertsMeta?["nflcom"]?.note { note("NFL.com", n) }
            if let n = comparison.expertsMeta?["rotowire"]?.note { note("RotoWire", n) }
            note("Caveat", "Each source is scored on the players it actually projects, so this is an approximate scoreboard, not a strictly paired test.")
        }
        .padding(.vertical, 4)
    }

    private func note(_ title: String, _ body: String) -> some View {
        (Text(title + ". ").font(.caption2.weight(.semibold)).foregroundColor(FFColor.textPrimary)
            + Text(body).font(.caption2).foregroundColor(FFColor.textSecondary))
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// One position's accuracy row set (our 4 models + 2 experts), best cell tinted.
struct ComparisonPositionGroup: View {
    let comparison: Comparison
    let subset: String
    let position: Position
    let metric: MetricKind

    var body: some View {
        let values = cmpSources.compactMap {
            comparison.cell(subset: subset, position: position.rawValue, source: $0.key)?.value(metric)
        }
        let best = metric.best(of: values)

        DisclosureGroup {
            ForEach(cmpSources) { source in
                let value = comparison.cell(subset: subset, position: position.rawValue, source: source.key)?.value(metric)
                HStack {
                    Circle().fill(source.color).frame(width: 8, height: 8)
                    Text(source.label).font(.caption).foregroundStyle(FFColor.textPrimary)
                    Spacer()
                    Text(value.map { metric.format($0) } ?? "—")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(valueIsBest(value, best) ? FFColor.accent : FFColor.textPrimary)
                        .fontWeight(valueIsBest(value, best) ? .bold : .regular)
                }
            }
        } label: {
            HStack {
                PositionBadge(position: position.rawValue)
                Spacer()
                if let best { Text(metric.format(best)).font(.caption.monospacedDigit()).foregroundStyle(FFColor.accent) }
            }
        }
    }
}

/// One position's residual-σ reliability rows (lower σ = steadier).
struct ReliabilityGroup: View {
    let comparison: Comparison
    let position: Position

    private struct Cell {
        let source: CmpSource
        let sigma: Double?
        let bias: Double?
        let n: Int?
        let totalsOnly: Bool
    }

    private func resolve(_ source: CmpSource) -> Cell {
        if source.isModel {
            let m = comparison.modelReliability(position: position.rawValue, model: source.key)
            return Cell(source: source, sigma: m?.sigma, bias: m?.bias, n: m?.n, totalsOnly: false)
        }
        let cell = comparison.expertReliability?.positions[position.rawValue]?[source.key] ?? nil
        let season = cell?.perSeason?["2025"]
        return Cell(source: source, sigma: season?.sigma, bias: season?.bias, n: season?.n, totalsOnly: cell?.totalsOnly ?? false)
    }

    var body: some View {
        let cells = cmpSources.map(resolve)
        let best = cells.compactMap(\.sigma).min()

        DisclosureGroup {
            ForEach(cells, id: \.source.id) { cell in
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Circle().fill(cell.source.color).frame(width: 8, height: 8)
                        Text(cell.source.label).font(.caption).foregroundStyle(FFColor.textPrimary)
                        if cell.totalsOnly { Text("totals-only").font(.caption2).foregroundStyle(FFColor.textMuted) }
                        Spacer()
                        Text(cell.sigma.map { Fmt.num($0, 2) } ?? "—")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(valueIsBest(cell.sigma, best) ? FFColor.accent : FFColor.textPrimary)
                            .fontWeight(valueIsBest(cell.sigma, best) ? .bold : .regular)
                    }
                    if let bias = cell.bias, let n = cell.n {
                        Text("bias \(Fmt.delta(bias)) · n=\(n)").font(.caption2).foregroundStyle(FFColor.textMuted)
                    }
                }
            }
        } label: {
            HStack {
                PositionBadge(position: position.rawValue)
                Spacer()
                if let best { Text("σ \(Fmt.num(best, 2))").font(.caption.monospacedDigit()).foregroundStyle(FFColor.accent) }
            }
        }
    }
}
