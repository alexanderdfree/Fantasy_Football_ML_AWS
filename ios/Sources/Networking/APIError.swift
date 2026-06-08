import Foundation

enum APIError: Error, LocalizedError {
    case invalidURL
    case http(Int)
    case decoding(String)
    case transport(String)

    /// `/api/snapshot` 404 → fall back to live predictions.
    var isNotFound: Bool {
        if case .http(404) = self { return true }
        return false
    }

    /// `/api/upcoming_week` 503 → the artifact is still building ("warming").
    var isWarming: Bool {
        if case .http(503) = self { return true }
        return false
    }

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid request URL."
        case let .http(code): return "Server returned an error (\(code))."
        case let .decoding(detail): return "Couldn't read the server response.\n\(detail)"
        case let .transport(message): return message
        }
    }
}
