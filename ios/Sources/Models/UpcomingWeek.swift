import Foundation

/// `/api/upcoming_week` artifact. Three shapes, all decoded by this one struct:
///   - available: `{available:true, week_label, scoring:{ppr:[…],…}, …}`
///   - offseason: `{available:false, reason:"offseason"|"no_slate"|"no_roster"}`
///   - warming:   `{status:"warming"}` (HTTP 503 — artifact not built yet)
struct UpcomingWeek: Codable, Sendable {
    let available: Bool?
    let reason: String?
    let status: String?
    let season: Int?
    let week: Int?
    let weekLabel: String?
    let generatedAt: String?
    let noActuals: Bool?
    let positions: [String]?
    let degradedPositions: [String]?
    let scoring: [String: [UpcomingPlayer]]?

    func players(_ format: ScoringFormat) -> [UpcomingPlayer] { scoring?[format.rawValue] ?? [] }

    enum CodingKeys: String, CodingKey {
        case available, reason, status, season, week, positions, scoring
        case weekLabel = "week_label"
        case generatedAt = "generated_at"
        case noActuals = "no_actuals"
        case degradedPositions = "degraded_positions"
    }
}

struct UpcomingPlayer: Codable, Identifiable, Sendable {
    let playerID: String
    let name: String
    let position: String
    let team: String
    let opponent: String?
    let isHome: Int?
    let spreadLine: Double?
    let totalLine: Double?
    let impliedTeamTotal: Double?
    let actual: Double? // always null (no games played)
    let ridgePred: Double?
    let nnPred: Double?
    let attnNNPred: Double?
    let lgbmPred: Double?
    let headshot: String

    var id: String { playerID }
    var positionEnum: Position? { Position(rawValue: position) }

    func prediction(for model: PredictionModel) -> Double? {
        switch model {
        case .ridge: return ridgePred
        case .nn: return nnPred
        case .attnNN: return attnNNPred
        case .lgbm: return lgbmPred
        }
    }

    /// Default sort key — best available projection (Attn NN preferred), mirrors
    /// `upcomingProjection` in app.js.
    var bestProjection: Double? { attnNNPred ?? lgbmPred ?? ridgePred ?? nnPred }

    var matchupLabel: String {
        guard let opponent, !opponent.isEmpty else { return "—" }
        return isHome == 1 ? "vs \(opponent)" : "@ \(opponent)"
    }

    enum CodingKeys: String, CodingKey {
        case name, position, team, opponent, actual, headshot
        case playerID = "player_id"
        case isHome = "is_home"
        case spreadLine = "spread_line"
        case totalLine = "total_line"
        case impliedTeamTotal = "implied_team_total"
        case ridgePred = "ridge_pred"
        case nnPred = "nn_pred"
        case attnNNPred = "attn_nn_pred"
        case lgbmPred = "lgbm_pred"
    }
}
