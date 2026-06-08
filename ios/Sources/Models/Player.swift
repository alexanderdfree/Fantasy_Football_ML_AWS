import Foundation

/// One player-week prediction row (`/api/predictions`, `/api/snapshot`).
struct Player: Codable, Identifiable, Hashable, Sendable {
    let playerID: String
    let name: String
    let position: String
    let team: String
    let week: Int
    let actual: Double?
    let ridgePred: Double?
    let nnPred: Double?
    let attnNNPred: Double?
    let lgbmPred: Double?
    let headshot: String

    /// player_id alone is not unique across weeks.
    var id: String { "\(playerID)-\(week)" }
    var positionEnum: Position? { Position(rawValue: position) }

    func prediction(for model: PredictionModel) -> Double? {
        switch model {
        case .ridge: return ridgePred
        case .nn: return nnPred
        case .attnNN: return attnNNPred
        case .lgbm: return lgbmPred
        }
    }

    /// Max across the available model predictions (min-projected-points filter).
    var maxPrediction: Double? {
        [ridgePred, nnPred, attnNNPred, lgbmPred].compactMap { $0 }.max()
    }

    enum CodingKeys: String, CodingKey {
        case playerID = "player_id"
        case name, position, team, week, actual
        case ridgePred = "ridge_pred"
        case nnPred = "nn_pred"
        case attnNNPred = "attn_nn_pred"
        case lgbmPred = "lgbm_pred"
        case headshot
    }
}

/// `/api/snapshot` — every player-week for all three scoring formats.
struct SnapshotResponse: Codable, Sendable {
    let generatedAt: String?
    let weeks: [Int]
    let degradedPositions: [String]
    let scoring: [String: [Player]]

    func players(_ format: ScoringFormat) -> [Player] { scoring[format.rawValue] ?? [] }

    /// Teams aren't carried in the payload — derive from rows (scoring-invariant).
    var teams: [String] {
        Set(players(.ppr).map(\.team).filter { !$0.isEmpty }).sorted()
    }

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case weeks
        case degradedPositions = "degraded_positions"
        case scoring
    }
}

/// `/api/predictions` — live fallback when the snapshot is absent.
struct PredictionsResponse: Codable, Sendable {
    let players: [Player]
    let total: Int
    let scoring: String
    let degradedPositions: [String]

    enum CodingKeys: String, CodingKey {
        case players, total, scoring
        case degradedPositions = "degraded_positions"
    }
}
