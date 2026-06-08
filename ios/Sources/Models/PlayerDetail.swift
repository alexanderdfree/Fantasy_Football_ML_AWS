import Foundation

/// `/api/player/<id>?scoring=` — header + week-by-week series for the chart.
struct PlayerDetail: Codable, Sendable {
    let playerID: String
    let name: String
    let position: String
    let team: String
    let headshot: String
    let weekly: [WeeklyPoint]
    let seasonAvg: Double?
    let seasonTotal: Double?
    let scoring: String

    var positionEnum: Position? { Position(rawValue: position) }

    struct WeeklyPoint: Codable, Sendable, Identifiable {
        let week: Int
        let actual: Double?
        let ridgePred: Double?
        let nnPred: Double?
        let attnNNPred: Double?
        let lgbmPred: Double?

        var id: Int { week }

        func prediction(for model: PredictionModel) -> Double? {
            switch model {
            case .ridge: return ridgePred
            case .nn: return nnPred
            case .attnNN: return attnNNPred
            case .lgbm: return lgbmPred
            }
        }

        enum CodingKeys: String, CodingKey {
            case week, actual
            case ridgePred = "ridge_pred"
            case nnPred = "nn_pred"
            case attnNNPred = "attn_nn_pred"
            case lgbmPred = "lgbm_pred"
        }
    }

    enum CodingKeys: String, CodingKey {
        case playerID = "player_id"
        case name, position, team, headshot, weekly
        case seasonAvg = "season_avg"
        case seasonTotal = "season_total"
        case scoring
    }
}
