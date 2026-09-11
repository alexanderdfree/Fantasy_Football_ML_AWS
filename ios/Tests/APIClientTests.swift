import XCTest
@testable import FFPredictor

private final class StubHTTP: URLProtocol {
    // Each test uses a separate URL host so concurrent tests cannot share replies.
    private static let lock = NSLock()
    private static var replies: [String: (Int, [String: String])] = [:]
    static func register(host: String, status: Int, headers: [String: String]) {
        lock.lock(); defer { lock.unlock() }
        replies[host] = (status, headers)
    }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock()
        let reply = Self.replies[request.url!.host!]!
        Self.lock.unlock()
        let response = HTTPURLResponse(url: request.url!, statusCode: reply.0, httpVersion: "HTTP/1.1", headerFields: reply.1)!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: Data("{\"status\":\"ok\"}".utf8))
        client?.urlProtocolDidFinishLoading(self)
    }
    override func stopLoading() {}
}

final class APIClientTests: XCTestCase {
    private func client(version: String?, status: Int = 200) -> APIClient {
        let host = UUID().uuidString.lowercased() + ".invalid"
        StubHTTP.register(host: host, status: status, headers: version.map { ["X-FFP-Contract-Version": $0] } ?? [:])
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [StubHTTP.self]
        return APIClient(base: URL(string: "https://" + host)!, session: URLSession(configuration: config))
    }

    func testCurrentAndLegacyVersionHeadersDecode() async throws {
        for version in [nil, "1.0", "1.2"] as [String?] {
            let health = try await client(version: version).get(.health, as: Health.self)
            XCTAssertEqual(health.status, "ok")
        }
    }

    func testUnknownMajorVersionIsNotDecoded() async {
        do {
            _ = try await client(version: "2.0").rawData(.health)
            XCTFail("Unsupported contract should fail")
        } catch let error as APIError {
            guard case .decoding = error else { return XCTFail("Expected contract decoding error") }
            XCTAssertFalse(error.isRetryable)
        } catch { XCTFail("Unexpected error: \(error)") }
    }

    func testNotFoundRetainsFallbackSemantics() async {
        do {
            _ = try await client(version: "1.0", status: 404).rawData(.snapshot)
            XCTFail("HTTP 404 should fail")
        } catch let error as APIError {
            XCTAssertTrue(error.isNotFound)
            XCTAssertFalse(error.isRetryable)
        } catch { XCTFail("Unexpected error: \(error)") }
    }
}
