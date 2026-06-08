import Foundation

/// `/api/position_details?scoring=` — top-level keyed by position ("QB"…"DST").
typealias PositionDetailsResponse = [String: PositionDetail]

struct PositionDetail: Codable, Sendable {
    let label: String
    let targets: [Target]
    let specificFeatures: [String]
    let architecture: ArchSummary
    let adjustments: String?
    let formula: String?
    let targetMetrics: [String: TargetMetricRow]?
    let nFeatures: Int?
    let nSamplesTest: Int?

    struct Target: Codable, Sendable, Identifiable {
        let key: String
        let label: String
        let formula: String
        var id: String { key }
    }

    struct ArchSummary: Codable, Sendable {
        let backbone: [Int]
        let headHidden: Int?

        enum CodingKeys: String, CodingKey {
            case backbone
            case headHidden = "head_hidden"
        }
    }

    /// `target_metrics[key]` — per-model MAE for one raw-stat target.
    /// The "total" row carries no `unit`.
    struct TargetMetricRow: Codable, Sendable {
        let ridgeMAE: Double?
        let nnMAE: Double?
        let attnNNMAE: Double?
        let lgbmMAE: Double?
        let unit: String?

        func mae(for model: PredictionModel) -> Double? {
            switch model {
            case .ridge: return ridgeMAE
            case .nn: return nnMAE
            case .attnNN: return attnNNMAE
            case .lgbm: return lgbmMAE
            }
        }

        enum CodingKeys: String, CodingKey {
            case ridgeMAE = "ridge_mae"
            case nnMAE = "nn_mae"
            case attnNNMAE = "attn_nn_mae"
            case lgbmMAE = "lgbm_mae"
            case unit
        }
    }

    enum CodingKeys: String, CodingKey {
        case label, targets, architecture, adjustments, formula
        case specificFeatures = "specific_features"
        case targetMetrics = "target_metrics"
        case nFeatures = "n_features"
        case nSamplesTest = "n_samples_test"
    }
}
