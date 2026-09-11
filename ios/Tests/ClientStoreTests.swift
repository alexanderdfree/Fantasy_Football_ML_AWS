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

private actor ControlledAPI: APIProviding {
    var requests: [Endpoint] = []
    var pending: [CheckedContinuation<Data, Error>] = []
    func rawData(_ endpoint: Endpoint) async throws -> Data {
        requests.append(endpoint)
        return try await withCheckedThrowingContinuation { pending.append($0) }
    }
    func waitForRequests(_ count: Int) async {
        while requests.count < count { await Task.yield() }
    }
    func complete(_ index: Int, _ result: Result<Data, APIError>) {
        switch result {
        case .success(let data): pending[index].resume(returning: data)
        case .failure(let error): pending[index].resume(throwing: error)
        }
    }
    func scoring(_ index: Int) -> String? {
        requests[index].query.first { $0.name == "scoring" }?.value
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

    func testOlderSnapshotCannotReplaceNewerResponseOrSavedBytes() async throws {
        let old = try fixture("snapshot")
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: old) as? [String: Any])
        object["weeks"] = [99]
        let latest = try JSONSerialization.data(withJSONObject: object)
        let api = ControlledAPI()
        let cache = MemorySnapshotCache()
        let store = SnapshotStore(api: api, cache: cache)
        let first = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(1)
        let second = Task { await store.hydrate(scoring: .standard) }
        await api.waitForRequests(2)
        await api.complete(1, .success(latest))
        await second.value
        await api.complete(0, .success(old))
        await first.value
        XCTAssertEqual(store.weeks, [99])
        XCTAssertEqual(cache.data, latest)
        XCTAssertFalse(store.isStale)
    }

    func testOlderRefreshErrorCannotMarkNewSnapshotStale() async throws {
        let api = ControlledAPI()
        let store = SnapshotStore(api: api, cache: MemorySnapshotCache())
        let first = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(1)
        let second = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(2)
        await api.complete(1, .success(try fixture("snapshot")))
        await second.value
        await api.complete(0, .failure(.http(500)))
        await first.value
        XCTAssertFalse(store.isStale)
        XCTAssertNil(store.errorMessage)
    }

    func testPendingSnapshotFallbackUsesLatestSelectedScoring() async throws {
        let api = ControlledAPI()
        let store = SnapshotStore(api: api, cache: MemorySnapshotCache(try fixture("snapshot")))
        let hydration = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(1)
        await store.ensureLive(.standard)
        await api.complete(0, .failure(.http(404)))
        await api.waitForRequests(2)
        let scoring = await api.scoring(1)
        XCTAssertEqual(scoring, "standard")
        let live = try fixture("predictions_qb_w1")
        await api.complete(1, .success(live))
        await hydration.value
        XCTAssertFalse(store.players(.standard).isEmpty)
        XCTAssertTrue(store.players(.ppr).isEmpty)
        XCTAssertFalse(store.usingSnapshot)
    }

    func testNewLiveGenerationDoesNotReuseAnotherFormatsOlderRows() async throws {
        let live = try fixture("predictions_qb_w1")
        let api = ScriptedAPI([.failure(.http(404)), .success(live), .failure(.http(404)), .success(live)])
        let store = SnapshotStore(api: api, cache: MemorySnapshotCache())
        await store.hydrate(scoring: .ppr)
        XCTAssertFalse(store.players(.ppr).isEmpty)
        await store.hydrate(scoring: .standard)
        XCTAssertFalse(store.players(.standard).isEmpty)
        XCTAssertTrue(store.players(.ppr).isEmpty)
    }

    func testInactiveScoringFailureCannotInvalidateCurrentLiveRows() async throws {
        let api = ControlledAPI()
        let store = SnapshotStore(api: api, cache: MemorySnapshotCache())
        let hydration = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(1)
        await api.complete(0, .failure(.http(404)))
        await api.waitForRequests(2)
        let standard = Task { await store.ensureLive(.standard) }
        await api.waitForRequests(3)
        await api.complete(2, .success(try fixture("predictions_qb_w1")))
        await standard.value
        await api.complete(1, .failure(.http(500)))
        await hydration.value
        XCTAssertFalse(store.players(.standard).isEmpty)
        XCTAssertFalse(store.isStale)
        XCTAssertNil(store.errorMessage)
    }

    func testExistingLiveModeResumesSelectedFormatAfterSnapshotRefreshFailure() async throws {
        let api = ControlledAPI()
        let store = SnapshotStore(api: api, cache: MemorySnapshotCache())
        let first = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(1)
        await api.complete(0, .failure(.http(404)))
        await api.waitForRequests(2)
        let live = try fixture("predictions_qb_w1")
        await api.complete(1, .success(live))
        await first.value
        let refresh = Task { await store.hydrate(scoring: .ppr) }
        await api.waitForRequests(3)
        await store.ensureLive(.standard)
        await api.complete(2, .failure(.http(503)))
        await api.waitForRequests(4)
        let scoring = await api.scoring(3)
        XCTAssertEqual(scoring, "standard")
        await api.complete(3, .success(live))
        await refresh.value
        XCTAssertFalse(store.players(.standard).isEmpty)
        XCTAssertFalse(store.isStale)
        XCTAssertNil(store.errorMessage)
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
