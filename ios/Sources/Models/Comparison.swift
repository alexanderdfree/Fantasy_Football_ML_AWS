import Foundation

/// `/api/comparison` — model and expert accuracy, with the server's scoring,
/// cohort, and coverage contract. Metadata stays optional for cached older APIs.
struct Comparison: Codable, Sendable {
    let scoring: String
    let modelSource: String
    let generatedAt: String?
    let topN: Int?
    let top12N: Int?
    let weeklyTopN: Int?
    let sampleBasis: String?
    let actualBasis: String?
    let scoringComponents: [String: [String]]?
    let excludedSources: [String: [String: String]]?
    let excludedComponents: [String: [String: String]]?
    let cohortDefinitions: [String: String]?
    let coverage: [String: [String: Coverage]]?
    /// subset -> POS -> source ("ridge"…"espn") -> cell.
    let subsets: [String: [String: [String: MetricCell?]]]
    /// POS -> model prefix -> {bias,mae,n,sigma}. POS value may be null.
    let modelReliability: [String: [String: ReliabilityBlock]?]?
    let expertReliability: ExpertReliability?
    let intervals: Intervals?
    let expertsMeta: [String: ExpertMeta]?

    var isUnavailable: Bool { modelSource == "unavailable" }

    var displayedSubsets: [String] {
        let order = ["weekly_reference_top24", "all", "top30", "top12"]
        return order.filter { subsets[$0] != nil } + subsets.keys.filter { !order.contains($0) }.sorted()
    }

    /// The payload owns source membership; this list only sets familiar display order.
    func sourceKeys(subset: String, position: String) -> [String] {
        let order = ["ridge", "nn", "attn_nn", "lgbm", "nflcom", "rotowire", "espn"]
        let keys = Set((subsets[subset]?[position] ?? [:]).keys)
            .union(coverage?[subset]?[position]?.sources ?? [])
            .union(excludedSources?[position]?.keys.map { $0 } ?? [])
            .union(coverage?[subset]?[position]?.excludedSources?.keys.map { $0 } ?? [])
        return order.filter { keys.contains($0) } + keys.filter { !order.contains($0) }.sorted()
    }

    func subsetTitle(_ subset: String) -> String {
        switch subset {
        case "weekly_reference_top24": return "Expected starters · weekly top \(weeklyTopN ?? 24)"
        case "all": return "All players"
        case "top30": return "Season leaders · top \(topN ?? 30)"
        case "top12": return "Season leaders · top \(top12N ?? 12)"
        default: return subset.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    var sampleBasisDescription: String {
        switch sampleBasis {
        case "shared_player_weeks":
            return "Every displayed source for a position is scored on the same regular-season player-weeks. Missing forecasts are excluded; a forecast of zero is retained."
        case .some(let basis): return "Server sample basis: \(basis)."
        case .none: return "This response does not specify its sample basis or whether sources share the same player-weeks."
        }
    }

    var actualBasisDescription: String {
        switch actualBasis {
        case "shared_projected_components_v1", "shared_projected_components_v2":
            return "Predictions and actuals use the shared projected components listed for each position. Stats outside those lists are excluded from both sides; these scores differ from full fantasy totals."
        case .some(let basis): return "Server actual basis: \(basis)."
        case .none: return "This response does not specify which actual scoring components are included."
        }
    }

    func exclusionReason(subset: String, position: String, source: String) -> String? {
        coverage?[subset]?[position]?.excludedSources?[source] ?? excludedSources?[position]?[source]
    }

    func cell(subset: String, position: String, source: String) -> MetricCell? {
        (subsets[subset]?[position]?[source]) ?? nil
    }

    func modelReliability(position: String, model: String) -> ReliabilityBlock? {
        (modelReliability?[position] ?? nil)?[model]
    }

    struct MetricCell: Codable, Sendable {
        let mae: Double?
        let rmse: Double?
        let r2: Double?
        let n: Int?

        func value(_ metric: MetricKind) -> Double? {
            switch metric {
            case .mae: return mae
            case .rmse: return rmse
            case .r2: return r2
            }
        }
    }

    struct Coverage: Codable, Sendable {
        let status: String
        let reason: String?
        let n: Int?
        let cohortN: Int?
        let sources: [String]?
        let sourceN: [String: Int]?
        let actualBasis: String?
        let scoringComponents: [String]?
        let excludedSources: [String: String]?
        let referenceStatus: String?
        let missingReferenceWeeks: Int?

        var summary: String {
            if status == "unavailable" {
                return "Unavailable" + (reason.map { " · " + $0.replacingOccurrences(of: "_", with: " ") } ?? "")
            }
            let count = n.map { "\($0) shared player-weeks" } ?? "Sample size unavailable"
            let cohort = cohortN.map { " of \($0) cohort player-weeks" } ?? ""
            let partial = status == "partial" || referenceStatus == "partial" ? " · partial reference" : ""
            return count + cohort + partial
        }

        enum CodingKeys: String, CodingKey {
            case status, reason, n, sources
            case cohortN = "cohort_n"
            case sourceN = "source_n"
            case actualBasis = "actual_basis"
            case scoringComponents = "scoring_components"
            case excludedSources = "excluded_sources"
            case referenceStatus = "reference_status"
            case missingReferenceWeeks = "missing_reference_weeks"
        }
    }

    struct ReliabilityBlock: Codable, Sendable {
        let bias: Double?
        let mae: Double?
        let n: Int?
        let sigma: Double?
    }

    struct ExpertReliability: Codable, Sendable {
        let note: String?
        let positions: [String: [String: ExpertReliabilityCell?]]
        let seasons: [Int]?

        enum CodingKeys: String, CodingKey { case note, positions, seasons }
    }

    struct ExpertReliabilityCell: Codable, Sendable {
        let bias: Double?
        let mae: Double?
        let n: Int?
        let sigma: Double?
        let totalsOnly: Bool?
        let perSeason: [String: SeasonStat]?

        enum CodingKeys: String, CodingKey {
            case bias, mae, n, sigma
            case totalsOnly = "totals_only"
            case perSeason = "per_season"
        }
    }

    struct SeasonStat: Codable, Sendable {
        let bias: Double?
        let mae: Double?
        let n: Int?
        let sigma: Double?
    }

    struct ExpertMeta: Codable, Sendable {
        let label: String?
        let note: String?
        let seasons: JSONValue? // string "2025" or array
    }

    // MARK: Prediction intervals
    struct Intervals: Codable, Sendable {
        /// expert -> POS -> block (block may be null).
        let intervals: [String: [String: IntervalBlock?]]
        let nominalCoverage: Double?
        let evalSeasons: [Int]?
        let method: String?
        let sourcesMeta: [String: JSONValue]?

        func block(source: String, position: String) -> IntervalBlock? {
            (intervals[source]?[position]) ?? nil
        }

        enum CodingKeys: String, CodingKey {
            case intervals, method
            case nominalCoverage = "nominal_coverage"
            case evalSeasons = "eval_seasons"
            case sourcesMeta = "sources_meta"
        }
    }

    struct IntervalBlock: Codable, Sendable {
        let calibration: Calibration?
        let examples: [BandExample]?
        let totalsOnly: Bool?
        let fitSeasons: [Int]?
        let skipped: Bool?

        enum CodingKeys: String, CodingKey {
            case calibration, examples, skipped
            case totalsOnly = "totals_only"
            case fitSeasons = "fit_seasons"
        }
    }

    struct Calibration: Codable, Sendable {
        let coverage: Double
        let flag: String?
        let meanWidth: Double
        let nEval: Int?
        let nFit: Int?

        enum CodingKeys: String, CodingKey {
            case coverage, flag
            case meanWidth = "mean_width"
            case nEval = "n_eval"
            case nFit = "n_fit"
        }
    }

    struct BandExample: Codable, Sendable, Identifiable {
        let actual: Double
        let ceiling: Double
        let floor: Double
        let inBand: Bool
        let median: Double
        let playerID: String?
        let playerName: String
        let projection: Double
        let season: Int?
        let week: Int

        var id: String { "\(playerID ?? playerName)-\(week)" }

        enum CodingKeys: String, CodingKey {
            case actual, ceiling, floor, median, projection, season, week
            case inBand = "in_band"
            case playerID = "player_id"
            case playerName = "player_name"
        }
    }

    enum CodingKeys: String, CodingKey {
        case scoring, subsets, intervals
        case modelSource = "model_source"
        case generatedAt = "generated_at"
        case topN = "top_n"
        case top12N = "top12_n"
        case weeklyTopN = "weekly_top_n"
        case sampleBasis = "sample_basis"
        case actualBasis = "actual_basis"
        case scoringComponents = "scoring_components"
        case excludedSources = "excluded_sources"
        case excludedComponents = "excluded_components"
        case cohortDefinitions = "cohort_definitions"
        case coverage
        case modelReliability = "model_reliability"
        case expertReliability = "expert_reliability"
        case expertsMeta = "experts_meta"
    }
}
