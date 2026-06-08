import Foundation

/// `/api/metrics?scoring=` — keyed by model DISPLAY name
/// ("Ridge Regression" / "Neural Network" / "Attention NN" / "LightGBM").
struct MetricsResponse: Codable, Sendable {
    let byModel: [String: ModelMetrics]

    init(from decoder: Decoder) throws {
        byModel = try decoder.singleValueContainer().decode([String: ModelMetrics].self)
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        try c.encode(byModel)
    }

    func metrics(for model: PredictionModel) -> ModelMetrics? { byModel[model.metricsDisplayName] }
}

struct ModelMetrics: Codable, Sendable {
    let overall: MetricBlock?
    let byPosition: [PositionMetricRow]

    enum CodingKeys: String, CodingKey {
        case overall
        case byPosition = "by_position"
    }
}

/// `{mae, rmse, r2}` — `compute_metrics` output. r2 may be null (NaN-scrubbed).
struct MetricBlock: Codable, Sendable {
    let mae: Double?
    let rmse: Double?
    let r2: Double?

    func value(_ metric: MetricKind) -> Double? {
        switch metric {
        case .mae: return mae
        case .rmse: return rmse
        case .r2: return r2
        }
    }
}

struct PositionMetricRow: Codable, Sendable, Identifiable {
    let position: String
    let mae: Double?
    let rmse: Double?
    let r2: Double?
    let nSamples: Int?

    var id: String { position }

    func value(_ metric: MetricKind) -> Double? {
        switch metric {
        case .mae: return mae
        case .rmse: return rmse
        case .r2: return r2
        }
    }

    enum CodingKeys: String, CodingKey {
        case position, mae, rmse, r2
        case nSamples = "n_samples"
    }
}
