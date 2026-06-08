import Foundation

/// A typed API endpoint: a path (relative to the host root) plus query items.
/// Most routes live under `/api`; `/health` sits at the host root.
struct Endpoint {
    let path: String
    var query: [URLQueryItem] = []

    func url(base: URL) -> URL? {
        var comps = URLComponents(string: base.absoluteString + "/" + path)
        if !query.isEmpty { comps?.queryItems = query }
        return comps?.url
    }

    private static func scoring(_ s: ScoringFormat) -> [URLQueryItem] {
        [URLQueryItem(name: "scoring", value: s.rawValue)]
    }

    private static func encodePath(_ raw: String) -> String {
        raw.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? raw
    }

    // MARK: Static routes
    static let snapshot = Endpoint(path: "api/snapshot")
    static let weeks = Endpoint(path: "api/weeks")
    static let teams = Endpoint(path: "api/teams")
    static let upcomingWeek = Endpoint(path: "api/upcoming_week")
    static let modelArchitecture = Endpoint(path: "api/model_architecture")
    static let comparison = Endpoint(path: "api/comparison")
    static let benchmarkHistory = Endpoint(path: "api/benchmark_history")
    static let wikiIndex = Endpoint(path: "api/wiki/index")
    static let health = Endpoint(path: "health") // host root, not /api

    // MARK: Parameterized routes
    static func predictions(
        position: String, week: String, search: String,
        sort: String, order: String, scoring: ScoringFormat
    ) -> Endpoint {
        var q = [
            URLQueryItem(name: "position", value: position),
            URLQueryItem(name: "week", value: week),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "order", value: order),
            URLQueryItem(name: "scoring", value: scoring.rawValue),
        ]
        if !search.isEmpty { q.append(URLQueryItem(name: "search", value: search)) }
        return Endpoint(path: "api/predictions", query: q)
    }

    static func player(id: String, scoring: ScoringFormat) -> Endpoint {
        Endpoint(path: "api/player/\(encodePath(id))", query: Self.scoring(scoring))
    }

    static func breakdown(playerID: String, week: Int) -> Endpoint {
        Endpoint(path: "api/predictions/breakdown", query: [
            URLQueryItem(name: "player_id", value: playerID),
            URLQueryItem(name: "week", value: String(week)),
        ])
    }

    static func metrics(_ s: ScoringFormat) -> Endpoint {
        Endpoint(path: "api/metrics", query: scoring(s))
    }

    static func weeklyAccuracy(_ s: ScoringFormat) -> Endpoint {
        Endpoint(path: "api/weekly_accuracy", query: scoring(s))
    }

    static func positionDetails(_ s: ScoringFormat) -> Endpoint {
        Endpoint(path: "api/position_details", query: scoring(s))
    }

    static func wikiDoc(slug: String) -> Endpoint {
        Endpoint(path: "api/wiki/\(encodePath(slug))")
    }
}
