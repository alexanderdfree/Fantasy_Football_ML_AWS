import XCTest
@testable import FFPredictor

private actor ScriptedAPI: APIProviding {
    var results: [Result<Data, APIError>]
    var paths: [String] = []
    init(_ results: [Result<Data, APIError>]) { self.results = results }
    func rawData(_ endpoint: Endpoint) async throws -> Data {
        paths.append(endpoint.path)
        guard !results.isEmpty else { throw APIError.transport("Unexpected request") }
        return try results.removeFirst().get()
    }
}

private final class MemorySnapshotCache: SnapshotCaching {
    var data: Data?
    init(_ data: Data? = nil) { self.data = data }
    func load() -> SnapshotResponse? { data.flatMap { try? JSONDecoder().decode(SnapshotResponse.self, from: $0) } }
    func save(_ data: Data) { self.data = data }
}

@MainActor
final class ClientStoreTests: XCTestCase {
    private func fixture(_ name: String) throws -> Data {
        let url = try XCTUnwrap(Bundle(for: Self.self).url(forResource: name, withExtension: "json"))
        return try Data(contentsOf: url)
    }

    func testComparisonRetryAfterFailureAndReuseAfterSuccess() async throws {
        let api = ScriptedAPI([.failure(.http(500)), .success(try fixture("comparison_current"))])
        let store = ComparisonStore(api: api)
        await store.load()
        XCTAssertNotNil(store.state.errorMessage)
        await store.load()
        XCTAssertEqual(store.state.value?.sampleBasis, "shared_player_weeks")
        await store.load()
        let paths = await api.paths
        XCTAssertEqual(paths, ["api/comparison", "api/comparison"])
    }

    func testFailedRefreshRetainsSavedSnapshotAndMarksItStale() async throws {
        let bytes = try fixture("snapshot")
        let cache = MemorySnapshotCache(bytes)
        let api = ScriptedAPI([.failure(.transport("Offline"))])
        let store = SnapshotStore(api: api, cache: cache)
        await store.hydrate(scoring: .ppr)
        XCTAssertTrue(store.hasData)
        XCTAssertTrue(store.usingSnapshot)
        XCTAssertTrue(store.isStale)
        XCTAssertFalse(store.players(.standard).isEmpty)
        XCTAssertEqual(cache.data, bytes)
        XCTAssertEqual(store.errorMessage, "Offline")
    }

    func testSnapshot404ReplacesSavedRowsWithLiveRows() async throws {
        let cache = MemorySnapshotCache(try fixture("snapshot"))
        let live = try fixture("predictions_qb_w1")
        let api = ScriptedAPI([.failure(.http(404)), .success(live)])
        let store = SnapshotStore(api: api, cache: cache)
        await store.hydrate(scoring: .ppr)
        let expected = try JSONDecoder().decode(PredictionsResponse.self, from: live)
        XCTAssertEqual(store.players(.ppr), expected.players)
        XCTAssertEqual(store.weeks, Set(expected.players.map(\.week)).sorted())
        XCTAssertEqual(store.teams, Set(expected.players.map(\.team).filter { !$0.isEmpty }).sorted())
        XCTAssertFalse(store.usingSnapshot)
        XCTAssertFalse(store.isStale)
        XCTAssertNil(store.errorMessage)
        let paths = await api.paths
        XCTAssertEqual(paths, ["api/snapshot", "api/predictions"])
    }

    func testFailedLiveFallbackKeepsSavedRows() async throws {
        let cache = MemorySnapshotCache(try fixture("snapshot"))
        let api = ScriptedAPI([.failure(.http(404)), .failure(.http(503))])
        let store = SnapshotStore(api: api, cache: cache)
        await store.hydrate(scoring: .ppr)
        XCTAssertTrue(store.usingSnapshot)
        XCTAssertTrue(store.isStale)
        XCTAssertFalse(store.players(.ppr).isEmpty)
    }

    func testSuccessfulRefreshClearsStaleStateAndPersistsBytes() async throws {
        let bytes = try fixture("snapshot")
        let cache = MemorySnapshotCache()
        let api = ScriptedAPI([.failure(.http(500)), .success(bytes)])
        let store = SnapshotStore(api: api, cache: cache)
        await store.hydrate(scoring: .ppr)
        XCTAssertNotNil(store.errorMessage)
        await store.hydrate(scoring: .ppr)
        XCTAssertFalse(store.isStale)
        XCTAssertNil(store.errorMessage)
        XCTAssertEqual(cache.data, bytes)
    }

    func testWarmingAndFailureAreDifferentUpcomingStates() async {
        let api = ScriptedAPI([.failure(.http(503)), .failure(.http(500))])
        let store = UpcomingStore(api: api)
        await store.load()
        guard case .warming = store.state else { return XCTFail("Expected warming") }
        await store.load()
        guard case .failed = store.state else { return XCTFail("Expected an actual failure") }
    }
}
