import Foundation

/// `/health` — `{status: ok|degraded|unhealthy}`. `positions_loaded` is present
/// only in the degraded shape, so it's optional.
struct Health: Codable, Sendable {
    let status: String
    let positionsLoaded: [String]?
    let positionLoadErrors: [String: JSONValue]?

    var isOK: Bool { status == "ok" }
    var isDegraded: Bool { status == "degraded" }

    enum CodingKeys: String, CodingKey {
        case status
        case positionsLoaded = "positions_loaded"
        case positionLoadErrors = "position_load_errors"
    }
}
