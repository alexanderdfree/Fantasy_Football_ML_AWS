import Foundation

/// Thin async/await client over URLSession. URLCache gives near-free response
/// caching (honors server Cache-Control); no third-party networking lib.
actor APIClient: APIProviding {
    static let shared = APIClient()

    private let session: URLSession
    private let base: URL

    init(base: URL = AppConfig.baseURL, session: URLSession? = nil) {
        let cfg = URLSessionConfiguration.default
        cfg.requestCachePolicy = .useProtocolCachePolicy
        cfg.urlCache = URLCache(memoryCapacity: 16 * 1024 * 1024, diskCapacity: 128 * 1024 * 1024)
        cfg.timeoutIntervalForRequest = 45
        cfg.timeoutIntervalForResource = 60
        cfg.waitsForConnectivity = false
        self.session = session ?? URLSession(configuration: cfg)
        self.base = base
    }

    /// GET raw bytes (used for the upcoming-week endpoint, which discriminates
    /// its payload shape, and for snapshot persistence).
    func rawData(_ endpoint: Endpoint) async throws -> Data {
        var delayNs: UInt64 = 500_000_000
        for attempt in 0..<3 {
            do {
                return try await rawDataOnce(endpoint)
            } catch let error as APIError where error.isRetryable && attempt < 2 {
                try await Task.sleep(nanoseconds: delayNs)
                delayNs *= 2
            }
        }
        return try await rawDataOnce(endpoint)
    }

    private func rawDataOnce(_ endpoint: Endpoint) async throws -> Data {
        guard let url = endpoint.url(base: base) else { throw APIError.invalidURL }
        do {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.transport("No HTTP response.")
            }
            guard (200..<300).contains(http.statusCode) else {
                throw APIError.http(http.statusCode)
            }
            if let version = http.value(forHTTPHeaderField: "X-FFP-Contract-Version"),
               version.split(separator: ".").first != "1" {
                throw APIError.decoding("Unsupported API contract version: \(version)")
            }
            return data
        } catch let error as APIError {
            throw error
        } catch let error as URLError {
            throw APIError.transport(error.localizedDescription, retryable: Self.isRetryable(error))
        }
    }

    nonisolated private static func isRetryable(_ error: URLError) -> Bool {
        switch error.code {
        case .timedOut, .networkConnectionLost, .cannotFindHost, .cannotConnectToHost, .dnsLookupFailed:
            return true
        default:
            return false
        }
    }
}
