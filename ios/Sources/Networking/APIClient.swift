import Foundation

/// Thin async/await client over URLSession. URLCache gives near-free response
/// caching (honors server Cache-Control); no third-party networking lib.
actor APIClient {
    static let shared = APIClient()

    private let session: URLSession
    private let decoder = JSONDecoder()
    private let base: URL

    init(base: URL = AppConfig.baseURL) {
        let cfg = URLSessionConfiguration.default
        cfg.requestCachePolicy = .useProtocolCachePolicy
        cfg.urlCache = URLCache(memoryCapacity: 16 * 1024 * 1024, diskCapacity: 128 * 1024 * 1024)
        cfg.timeoutIntervalForRequest = 20
        cfg.waitsForConnectivity = false
        self.session = URLSession(configuration: cfg)
        self.base = base
    }

    /// GET + JSON-decode. Throws `APIError` (incl. `.http(404)` / `.http(503)`
    /// callers branch on for snapshot-fallback / upcoming-week "warming").
    func get<T: Decodable>(_ endpoint: Endpoint, as type: T.Type = T.self) async throws -> T {
        let data = try await rawData(endpoint)
        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            throw APIError.decoding(String(describing: error))
        }
    }

    /// GET raw bytes (used for the upcoming-week endpoint, which discriminates
    /// its payload shape, and for snapshot persistence).
    func rawData(_ endpoint: Endpoint) async throws -> Data {
        guard let url = endpoint.url(base: base) else { throw APIError.invalidURL }
        do {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse else {
                throw APIError.transport("No HTTP response.")
            }
            guard (200..<300).contains(http.statusCode) else {
                throw APIError.http(http.statusCode)
            }
            return data
        } catch let error as APIError {
            throw error
        } catch let error as URLError {
            throw APIError.transport(error.localizedDescription)
        }
    }
}
