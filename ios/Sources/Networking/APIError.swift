import Foundation

enum APIError: Error, LocalizedError, Sendable {
    case invalidURL
    case http(Int)
    case decoding(String)
    case transport(String, retryable: Bool = false)

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

    var isRetryable: Bool {
        switch self {
        case let .http(code): return [502, 503, 504].contains(code)
        case let .transport(_, retryable): return retryable
        case .invalidURL, .decoding: return false
        }
    }

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid request URL."
        case let .http(code): return "Server returned an error (\(code))."
        case let .decoding(detail): return "Couldn't read the server response.\n\(detail)"
        case let .transport(message, _): return message
        }
    }
}
