import Foundation

/// The store boundary: production URLSession and deterministic test clients
/// supply the same bytes. Decoding remains shared, including error semantics.
protocol APIProviding: Sendable {
    func rawData(_ endpoint: Endpoint) async throws -> Data
}

extension APIProviding {
    func get<T: Decodable>(_ endpoint: Endpoint, as type: T.Type = T.self) async throws -> T {
        let data = try await rawData(endpoint)
        do { return try JSONDecoder().decode(T.self, from: data) }
        catch { throw APIError.decoding(String(describing: error)) }
    }
}
