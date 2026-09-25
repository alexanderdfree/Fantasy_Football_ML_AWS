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

    /// Default sort key: the first available projection in this position's
    /// served-model chain, the same chain the web board ranks by and the
    /// Comparison tab's verdict grades (`ServedModelChain`).
    var bestProjection: Double? {
        for model in ServedModelChain.chain(for: position) {
            if let value = prediction(for: model) { return value }
        }
        return nil
    }

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

/// Per-position head selection (ADR-0003): the model the Next Week board ranks
/// first, then the fallbacks used when that forecast is missing. Mirrors
/// `SERVED_MODEL_CHAIN` in `src/contracts/api.py`; the Comparison payload's
/// `served_model` names each chain's first entry and `DecodingTests` pins the two
/// in step, so a chain change lands on the web board, the tab and this board.
enum ServedModelChain {
    static let chains: [String: [PredictionModel]] = [
        "QB": [.attnNN, .lgbm, .nn],
        "RB": [.lgbm, .attnNN, .nn],
        "WR": [.lgbm, .attnNN, .nn],
        "TE": [.attnNN, .lgbm, .nn],
        "K": [.ridge, .attnNN, .lgbm, .nn],
        "DST": [.attnNN, .lgbm, .nn],
    ]
    private static let fallback: [PredictionModel] = [.attnNN, .lgbm, .nn]

    static func chain(for position: String) -> [PredictionModel] {
        chains[position.uppercased()] ?? fallback
    }
}
