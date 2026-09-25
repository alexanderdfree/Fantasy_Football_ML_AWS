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
    /// Set when the graded season is the one model changes were compared on.
    let evaluationSeasonNote: String?
    /// POS -> the model the Next Week board ranks first; its interval is the verdict.
    let servedModel: [String: String]?
    /// Backtest inputs that were not all pregame, disclosed beside every verdict.
    let informationSetNote: String?
    let coverage: [String: [String: Coverage]]?
    /// subset -> POS -> source ("ridge"…"espn") -> cell.
    let subsets: [String: [String: [String: MetricCell?]]]
    /// POS -> model prefix -> {bias,mae,n,sigma}. POS value may be null.
    let modelReliability: [String: [String: ReliabilityBlock]?]?
    let expertReliability: ExpertReliability?
    let intervals: Intervals?
    let expertsMeta: [String: ExpertMeta]?

    var isUnavailable: Bool { modelSource == "unavailable" }

    /// Cohorts selected on outcomes or on a graded expert's own forecasts: cells
    /// and bias only, never a verdict, whatever the payload carries.
    static let noVerdictSubsets: Set<String> = ["top12", "top30", "weekly_reference_top24"]

    var displayedSubsets: [String] {
        let order = ["weekly_depth_starters", "all", "elite_top24", "weekly_reference_top24", "top30", "top12"]
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
        case "weekly_depth_starters": return "Expected starters · pregame depth chart"
        case "elite_top24": return "Prior-season elite · top \(weeklyTopN ?? 24)"
        case "weekly_reference_top24": return "Expert reference · weekly top \(weeklyTopN ?? 24)"
        case "all": return "All players"
        case "top30": return "Season leaders · top \(topN ?? 30)"
        case "top12": return "Season leaders · top \(top12N ?? 12)"
        default: return subset.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    var sampleBasisDescription: String {
        switch sampleBasis {
        case "shared_player_weeks":
            return "Every displayed source for a position is scored on the same regular-season player-weeks. Missing forecasts are excluded. A projected zero is retained, but a provider row with every published stat at zero is an unprojected placeholder and counts as missing."
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
        /// Mean prediction − actual (positive over-predicts). Shown, never ranked.
        let bias: Double?
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
        /// Paired, player-clustered bootstrap: the served model and the best of four
        /// against the best expert. Hindsight cohorts carry `not_applicable`.
        let uncertainty: Uncertainty?

        /// The served block when it can decide a row.
        private var served: ServedGap? {
            guard let served = uncertainty?.servedModel, served.status == "available" else { return nil }
            return served
        }

        /// The winning group ("models" / "experts") under the served model's
        /// both-metrics rule, or nil for a tie, a hindsight cohort, a snapshot
        /// without a usable served block, or no interval. Best of four never decides.
        var decidedWinner: String? {
            guard uncertainty?.status == "available", let winner = served?.winner, winner != "tie" else { return nil }
            return winner
        }

        /// The served model's key when it decided the row for the models, so only
        /// its own cell is highlighted.
        var decidedModel: String? { decidedWinner == "models" ? served?.model : nil }

        private static func label(_ winner: String?, _ verdict: String?, _ key: String) -> String {
            // A decided row needs both metrics; one decided metric alone stays a tie.
            winner == "models" ? "Models ahead"
                : winner == "experts" ? "Experts ahead"
                : verdict == "models" ? "≈ tie (models ahead on \(key) only)"
                : verdict == "experts" ? "≈ tie (experts ahead on \(key) only)" : "≈ tie"
        }

        /// Lowercase only the leading group word; the metric abbreviation keeps its case.
        private static func asContext(_ text: String) -> String {
            text.hasPrefix("≈") ? text : text.lowercased()
        }

        private static func interval(_ delta: Double, _ ci: [Double], _ key: String) -> String {
            let signed = { (value: Double) in String(format: "%+.2f", value) }
            return "\(signed(delta)) [\(signed(ci[0])), \(signed(ci[1]))] \(key)"
        }

        /// The row verdict for the shown metric: "Models ahead · Attention NN − best
        /// expert −0.08 [−0.23, +0.09] MAE". `modelLabel` renders a model key. A
        /// snapshot without a served block, or whose served model was not graded,
        /// gets a plain no-verdict line; best of four never becomes the verdict.
        func verdict(_ metric: MetricKind, modelLabel: (String) -> String = { $0 }) -> String? {
            guard let uncertainty, uncertainty.status == "available" else { return nil }
            let key = metric == .mae ? "MAE" : "RMSE"
            guard let block = uncertainty.servedModel else { return "No served-model verdict in this snapshot" }
            guard let served, let gap = metric == .mae ? served.mae : served.rmse, let model = served.model,
                  let delta = gap.minusBestExpert, let ci = gap.ci, ci.count == 2 else {
                return "No verdict · \(modelLabel(block.model ?? "the served model")) not graded on these rows"
            }
            let fallback = served.fallback == true
                ? " (next on the board; \(modelLabel(served.requested ?? "")) not graded)" : ""
            return "\(Self.label(served.winner, gap.verdict, key)) · \(modelLabel(model)) − best expert "
                + Self.interval(delta, ci, key) + fallback
        }

        /// Context beneath the verdict: the best-of-four gap, which gives the model
        /// family four draws and therefore never decides a row.
        func familyVerdict(_ metric: MetricKind) -> String? {
            guard let uncertainty, uncertainty.status == "available",
                  let gap = metric == .mae ? uncertainty.mae : uncertainty.rmse,
                  let delta = gap.bestModelMinusBestExpert, let ci = gap.ci, ci.count == 2 else { return nil }
            let key = metric == .mae ? "MAE" : "RMSE"
            return "best of four − best expert " + Self.interval(delta, ci, key) + " · "
                + Self.asContext(Self.label(uncertainty.winner, gap.verdict, key))
        }

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
            case uncertainty
        }
    }

    /// R² on one common sample orders sources exactly as RMSE, so it uses the RMSE interval.
    struct Uncertainty: Codable, Sendable {
        let status: String
        let reason: String?
        let winner: String?
        let mae: Gap?
        let rmse: Gap?
        let servedModel: ServedGap?

        enum CodingKeys: String, CodingKey {
            case status, reason, winner, mae, rmse
            case servedModel = "served_model"
        }
    }

    /// The served model (the first graded model in the board's ranking chain)
    /// against the best expert. `requested` is the chain's first model; `fallback`
    /// is true when it had no graded forecasts and the next one was used.
    struct ServedGap: Codable, Sendable {
        let status: String
        let reason: String?
        let model: String?
        let requested: String?
        let fallback: Bool?
        let winner: String?
        let mae: ServedMetricGap?
        let rmse: ServedMetricGap?
    }

    struct ServedMetricGap: Codable, Sendable {
        let bestExpert: String?
        let minusBestExpert: Double?
        let ci: [Double]?
        let verdict: String?

        enum CodingKeys: String, CodingKey {
            case ci, verdict
            case bestExpert = "best_expert"
            case minusBestExpert = "minus_best_expert"
        }
    }

    struct Gap: Codable, Sendable {
        let bestModel: String?
        let bestExpert: String?
        let bestModelMinusBestExpert: Double?
        let ci: [Double]?
        let verdict: String?

        enum CodingKeys: String, CodingKey {
            case ci, verdict
            case bestModel = "best_model"
            case bestExpert = "best_expert"
            case bestModelMinusBestExpert = "best_model_minus_best_expert"
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
        case evaluationSeasonNote = "evaluation_season_note"
        case servedModel = "served_model"
        case informationSetNote = "information_set_note"
        case coverage
        case modelReliability = "model_reliability"
        case expertReliability = "expert_reliability"
        case expertsMeta = "experts_meta"
    }
}
