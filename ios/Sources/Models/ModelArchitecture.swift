import Foundation

/// `/api/model_architecture` — overview, training loop, per-position config.
struct ModelArchitecture: Codable, Sendable {
    let overview: Overview
    let trainingLoop: TrainingLoop
    let positions: [String: PositionArch]

    struct Overview: Codable, Sendable {
        let framework: String
        let device: String
        let dataSplits: String
        let ensemble: [String]

        enum CodingKeys: String, CodingKey {
            case framework, device, ensemble
            case dataSplits = "data_splits"
        }
    }

    struct TrainingLoop: Codable, Sendable {
        let optimizer: String
        let loss: String
        let gradientClip: String
        let featureScaling: String
        let earlyStopping: String
        let checkpoint: String

        enum CodingKeys: String, CodingKey {
            case optimizer, loss, checkpoint
            case gradientClip = "gradient_clip"
            case featureScaling = "feature_scaling"
            case earlyStopping = "early_stopping"
        }
    }

    struct PositionArch: Codable, Sendable {
        let targets: [String]
        let backboneLayers: [Int]
        let headHidden: Int?
        let headHiddenOverrides: [String: Int]?
        let dropout: Double?
        let lr: Double?
        let weightDecay: Double?
        let batchSize: Int?
        let epochs: Int?
        let patience: Int?
        let scheduler: String
        let attentionEnabled: Bool
        let lightgbmEnabled: Bool
        let featureCount: Int
        let features: [String: [String]]

        enum CodingKeys: String, CodingKey {
            case targets, dropout, lr, epochs, patience, scheduler, features
            case backboneLayers = "backbone_layers"
            case headHidden = "head_hidden"
            case headHiddenOverrides = "head_hidden_overrides"
            case weightDecay = "weight_decay"
            case batchSize = "batch_size"
            case attentionEnabled = "attention_enabled"
            case lightgbmEnabled = "lightgbm_enabled"
            case featureCount = "feature_count"
        }
    }
}
