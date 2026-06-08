import Foundation

/// `/api/comparison` — our four models vs experts (NFL.com, RotoWire). Always
/// PPR (server-pinned). Expert/reliability/interval cells are decoded
/// defensively (optional fields; nullable dict values).
struct Comparison: Codable, Sendable {
    let scoring: String
    let modelSource: String
    let generatedAt: String?
    let topN: Int?
    /// subset ("all"/"top30") -> POS -> source ("ridge"…"rotowire") -> cell.
    let subsets: [String: [String: [String: MetricCell?]]]
    /// POS -> model prefix -> {bias,mae,n,sigma}. POS value may be null.
    let modelReliability: [String: [String: ReliabilityBlock]?]?
    let expertReliability: ExpertReliability?
    let intervals: Intervals?
    let expertsMeta: [String: ExpertMeta]?

    var isUnavailable: Bool { modelSource == "unavailable" }

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
        case modelReliability = "model_reliability"
        case expertReliability = "expert_reliability"
        case expertsMeta = "experts_meta"
    }
}
