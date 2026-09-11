import SwiftUI

/// Display attributes for source IDs supplied by the comparison response.
struct CmpSource: Identifiable {
    let key: String
    let label: String
    let color: Color
    let isModel: Bool
    var id: String { key }

    static func resolve(_ key: String, comparison: Comparison) -> CmpSource {
        if let known = cmpSources.first(where: { $0.key == key }) { return known }
        return CmpSource(key: key, label: comparison.expertsMeta?[key]?.label ?? key,
                         color: FFColor.textSecondary, isModel: false)
    }
}

let cmpSources: [CmpSource] =
    PredictionModel.allCases.map { CmpSource(key: $0.bareKey, label: $0.shortLabel, color: $0.color, isModel: true) }
    + [
        CmpSource(key: "nflcom", label: "NFL.com", color: FFColor.textSecondary, isModel: false),
        CmpSource(key: "rotowire", label: "RotoWire", color: FFColor.textSecondary, isModel: false),
        CmpSource(key: "espn", label: "ESPN", color: FFColor.textSecondary, isModel: false),
    ]

private func valueIsBest(_ value: Double?, _ best: Double?) -> Bool {
    guard let value, let best else { return false }
    return abs(value - best) < 1e-9
}

/// Our models vs experts, preserving server scoring and cohort semantics.
struct ComparisonView: View {
    @State private var store = ComparisonStore()
    @State private var metric: MetricKind = .mae

    var body: some View {
        LoadStateView(state: store.state, retry: { Task { await store.load() } }) { comparison in
            List {
                ForEach(comparison.displayedSubsets, id: \.self) { subset in
                    Section {
                        ForEach(Position.displayOrder) { pos in
                            ComparisonPositionGroup(comparison: comparison, subset: subset, position: pos, metric: metric)
                        }
                    } header: {
                        Text(comparison.subsetTitle(subset))
                    } footer: {
                        if let definition = comparison.cohortDefinitions?[subset] {
                            Text(definition)
                        }
                    }
                    .listRowBackground(FFColor.bgSecondary)
                }

                if comparison.expertReliability != nil {
                    Section("Historical reliability — residual σ (2025)") {
                        ForEach(Position.displayOrder) { pos in
                            ReliabilityGroup(comparison: comparison, position: pos)
                        }
                    }
                    .listRowBackground(FFColor.bgSecondary)
                }

                if let intervals = comparison.intervals {
                    Section("Historical prediction intervals") {
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
            note("Scoring", ScoringFormat(rawValue: comparison.scoring)?.displayName ?? comparison.scoring)
            note("Actuals", comparison.actualBasisDescription)
            note("Coverage", comparison.sampleBasisDescription)
            if comparison.isUnavailable {
                note("Availability", "Model comparison data is unavailable.")
            }
            ForEach((comparison.expertsMeta ?? [:]).keys.sorted(), id: \.self) { key in
                if let text = comparison.expertsMeta?[key]?.note {
                    note(CmpSource.resolve(key, comparison: comparison).label, text)
                }
            }
        }
        .padding(.vertical, 4)
    }

    private func note(_ title: String, _ body: String) -> some View {
        (Text(title + ". ").font(.caption2.weight(.semibold)).foregroundColor(FFColor.textPrimary)
            + Text(body).font(.caption2).foregroundColor(FFColor.textSecondary))
            .fixedSize(horizontal: false, vertical: true)
    }
}

/// One position's accuracy rows, source coverage, and shared scoring components.
struct ComparisonPositionGroup: View {
    let comparison: Comparison
    let subset: String
    let position: Position
    let metric: MetricKind

    var body: some View {
        let sources = comparison.sourceKeys(subset: subset, position: position.rawValue)
            .map { CmpSource.resolve($0, comparison: comparison) }
        let coverage = comparison.coverage?[subset]?[position.rawValue]
        let components = coverage?.scoringComponents ?? comparison.scoringComponents?[position.rawValue]
        let values = sources.compactMap {
            comparison.cell(subset: subset, position: position.rawValue, source: $0.key)?.value(metric)
        }
        let best = metric.best(of: values)

        DisclosureGroup {
            Text(coverage?.summary ?? "Coverage metadata unavailable")
                .font(.caption2).foregroundStyle(FFColor.textSecondary)
            if let missing = coverage?.missingReferenceWeeks, missing > 0 {
                Text("Missing pregame reference for \(missing) evaluation weeks")
                    .font(.caption2).foregroundStyle(FFColor.textSecondary)
            }
            if let components, !components.isEmpty {
                Text("Scored components: " + components.map { $0.replacingOccurrences(of: "_", with: " ") }.joined(separator: ", "))
                    .font(.caption2).foregroundStyle(FFColor.textSecondary)
            }
            let excluded = comparison.excludedComponents?[position.rawValue] ?? [:]
            ForEach(excluded.keys.sorted(), id: \.self) { component in
                Text("Excluded " + component.replacingOccurrences(of: "_", with: " ") + ": " + (excluded[component] ?? ""))
                    .font(.caption2).foregroundStyle(FFColor.textSecondary)
            }
            ForEach(sources) { source in
                let value = comparison.cell(subset: subset, position: position.rawValue, source: source.key)?.value(metric)
                VStack(alignment: .leading, spacing: 2) {
                    HStack {
                        Circle().fill(source.color).frame(width: 8, height: 8)
                        Text(source.label).font(.caption).foregroundStyle(FFColor.textPrimary)
                        Spacer()
                        Text(value.map { metric.format($0) } ?? "—")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(valueIsBest(value, best) ? FFColor.accent : FFColor.textPrimary)
                            .fontWeight(valueIsBest(value, best) ? .bold : .regular)
                    }
                    if let reason = comparison.exclusionReason(subset: subset, position: position.rawValue, source: source.key) {
                        Text("Excluded: " + reason).font(.caption2).foregroundStyle(FFColor.textMuted)
                    } else if let count = coverage?.sourceN?[source.key] {
                        Text("\(count) forecasts before shared filtering").font(.caption2).foregroundStyle(FFColor.textMuted)
                    }
                }
            }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    PositionBadge(position: position.rawValue)
                    if let coverage {
                        Text(coverage.summary).font(.caption2).foregroundStyle(FFColor.textMuted)
                    }
                }
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
        // This archived block used 2025 held-out model residuals. Keep expert
        // residuals on that same season instead of substituting all-season data.
        let season = cell?.perSeason?["2025"]
        return Cell(source: source, sigma: season?.sigma, bias: season?.bias, n: season?.n, totalsOnly: cell?.totalsOnly ?? false)
    }

    var body: some View {
        let cells = cmpSources.filter { source in
            if source.isModel { return comparison.modelReliability(position: position.rawValue, model: source.key) != nil }
            return comparison.expertReliability?.positions[position.rawValue]?[source.key] != nil
        }.map(resolve)
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
