import Foundation

/// `/api/benchmark_history` — one row per CI training run (newest first).
struct BenchmarkHistory: Codable, Sendable {
    let repoSlug: String
    let rows: [Row]
    let targetLabels: [String: String]
    let targetUnits: [String: String]

    struct Row: Codable, Sendable, Identifiable {
        let timestamp: String?
        let gitHash: String?
        let prNumber: Int?
        let trainingSkipped: Bool
        let positions: [String]
        let ridge: [Pill]
        let nn: [Pill]
        let attnNN: [Pill]
        let lgbm: [Pill]
        let totalElapsedSec: Double?

        var id: String { gitHash ?? timestamp ?? "\(prNumber ?? -1)" }

        func pills(for model: PredictionModel) -> [Pill] {
            switch model {
            case .ridge: return ridge
            case .nn: return nn
            case .attnNN: return attnNN
            case .lgbm: return lgbm
            }
        }

        enum CodingKeys: String, CodingKey {
            case timestamp, positions, ridge, nn, lgbm
            case gitHash = "git_hash"
            case prNumber = "pr_number"
            case trainingSkipped = "training_skipped"
            case attnNN = "attn_nn"
            case totalElapsedSec = "total_elapsed_sec"
        }
    }

    /// One position's metrics within a model column for a run.
    struct Pill: Codable, Sendable {
        let position: String
        let mae: Double?
        let rmse: Double?
        let perTarget: [String: Double]?
        let perTargetRMSE: [String: Double]?

        func value(_ metric: MetricKind) -> Double? { metric == .rmse ? rmse : mae }
        func perTargetMap(_ metric: MetricKind) -> [String: Double]? {
            metric == .rmse ? perTargetRMSE : perTarget
        }

        enum CodingKeys: String, CodingKey {
            case position, mae, rmse
            case perTarget = "per_target"
            case perTargetRMSE = "per_target_rmse"
        }
    }

    enum CodingKeys: String, CodingKey {
        case rows
        case repoSlug = "repo_slug"
        case targetLabels = "target_labels"
        case targetUnits = "target_units"
    }
}
