import Foundation

/// `/api/predictions/breakdown?player_id&week` — per-raw-stat actual vs the four
/// models. Scoring-invariant; model keys are bare (`ridge`, not `ridge_pred`).
struct Breakdown: Codable, Sendable {
    let playerID: String
    let week: Int
    let position: String
    let components: [Component]
    let unavailableModels: [String]
    let unavailable: Bool

    struct Component: Codable, Sendable, Identifiable {
        let key: String
        let label: String
        let unit: String
        let actual: Double?
        let ridge: Double?
        let nn: Double?
        let attnNN: Double?
        let lgbm: Double?

        var id: String { key }

        func value(for model: PredictionModel) -> Double? {
            switch model {
            case .ridge: return ridge
            case .nn: return nn
            case .attnNN: return attnNN
            case .lgbm: return lgbm
            }
        }

        enum CodingKeys: String, CodingKey {
            case key, label, unit, actual, ridge, nn
            case attnNN = "attn_nn"
            case lgbm
        }
    }

    enum CodingKeys: String, CodingKey {
        case playerID = "player_id"
        case week, position, components
        case unavailableModels = "unavailable_models"
        case unavailable
    }
}
