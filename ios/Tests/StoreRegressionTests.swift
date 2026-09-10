import Foundation
import XCTest
@testable import FFPredictor

private final class FixtureURLProtocol: URLProtocol {
    struct Response {
        let status: Int
        let data: Data
        var delay: TimeInterval = 0
    }
    static var handler: ((URLRequest) throws -> Response)?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let response = try Self.handler!(request)
            let finish = {
                let http = HTTPURLResponse(url: self.request.url!, statusCode: response.status,
                                           httpVersion: nil, headerFields: nil)!
                self.client?.urlProtocol(self, didReceive: http, cacheStoragePolicy: .notAllowed)
                self.client?.urlProtocol(self, didLoad: response.data)
                self.client?.urlProtocolDidFinishLoading(self)
            }
            if response.delay > 0 {
                DispatchQueue.global().asyncAfter(deadline: .now() + response.delay, execute: finish)
            } else {
                finish()
            }
        } catch {
            client?.urlProtocol(self, didFailWithError: error)
        }
    }
    override func stopLoading() {}
}

final class StoreRegressionTests: XCTestCase {
    private func client() -> APIClient {
        let config = URLSessionConfiguration.ephemeral
        config.protocolClasses = [FixtureURLProtocol.self]
        return APIClient(base: URL(string: "https://fixture.invalid")!, session: URLSession(configuration: config))
    }

    private static func player(_ name: String, week: Int, actual: Double) -> Player {
        Player(playerID: "00-test", name: name, position: "QB", team: name, week: week,
               actual: actual, ridgePred: actual, nnPred: actual, attnNNPred: actual,
               lgbmPred: actual, headshot: "")
    }

    @MainActor
    func testLiveFallbackReplacesDiskSnapshotAndRefreshesFilters() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = SnapshotCache(directory: directory)
        let old = Self.player("old", week: 1, actual: 1)
        cache.save(try JSONEncoder().encode(SnapshotResponse(
            generatedAt: nil, weeks: [1], degradedPositions: ["WR"],
            scoring: ["ppr": [old], "standard": [old]]
        )))
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                return .init(status: 404, data: Data())
            }
            let format = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
                .queryItems!.first { $0.name == "scoring" }!.value!
            let response = PredictionsResponse(players: [Self.player(format, week: 2, actual: 20)],
                                               total: 1, scoring: format, degradedPositions: [])
            return .init(status: 200, data: try JSONEncoder().encode(response))
        }
        let store = SnapshotStore(api: client(), cache: cache)
        await store.hydrate(scoring: .ppr)
        XCTAssertEqual(store.players(.ppr).first?.name, "ppr")
        XCTAssertEqual(store.weeks, [2])
        XCTAssertEqual(store.teams, ["ppr"])
        XCTAssertFalse(store.usingSnapshot)
        XCTAssertTrue(store.degradedPositions.isEmpty)
        await store.ensureLive(.standard)
        XCTAssertEqual(store.players(.standard).first?.name, "standard")
    }

    @MainActor
    func testFailedLiveFallbackPreservesOfflineSnapshot() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = SnapshotCache(directory: directory)
        let old = Self.player("offline", week: 1, actual: 1)
        cache.save(try JSONEncoder().encode(SnapshotResponse(
            generatedAt: nil, weeks: [1], degradedPositions: [], scoring: ["ppr": [old]]
        )))
        FixtureURLProtocol.handler = { request in
            .init(status: request.url!.path == "/api/snapshot" ? 404 : 500, data: Data())
        }
        let store = SnapshotStore(api: client(), cache: cache)
        await store.hydrate(scoring: .ppr)
        XCTAssertEqual(store.players(.ppr).first?.name, "offline")
        XCTAssertTrue(store.hasData)
    }

    @MainActor
    func testLiveRefreshInvalidatesOtherScoringFormats() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var revision = 1.0
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                return .init(status: 404, data: Data())
            }
            let format = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
                .queryItems!.first { $0.name == "scoring" }!.value!
            let response = PredictionsResponse(players: [Self.player(format, week: 1, actual: revision)],
                                               total: 1, scoring: format, degradedPositions: [])
            return .init(status: 200, data: try JSONEncoder().encode(response))
        }
        let store = SnapshotStore(api: client(), cache: SnapshotCache(directory: directory))
        await store.hydrate(scoring: .ppr)
        await store.ensureLive(.standard)
        XCTAssertEqual(store.players(.standard).first?.actual, 1)
        revision = 2
        await store.hydrate(scoring: .ppr)
        await store.ensureLive(.standard)
        XCTAssertEqual(store.players(.ppr).first?.actual, 2)
        XCTAssertEqual(store.players(.standard).first?.actual, 2)
    }

    @MainActor
    func testLatestPlayerScoringRequestWins() async throws {
        try await verifyPlayerRequestRace(earlierStatus: 200)
    }

    @MainActor
    func testFreshSnapshotSupersedesPendingLiveResult() async throws {
        try await verifySnapshotWinsOverEarlierLive(status: 200)
    }

    @MainActor
    func testFreshSnapshotSupersedesPendingLiveFailure() async throws {
        try await verifySnapshotWinsOverEarlierLive(status: 500)
    }

    @MainActor
    private func verifySnapshotWinsOverEarlierLive(status: Int) async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let started = expectation(description: "earlier live request started")
        var snapshotCalls = 0
        let fresh = Self.player("fresh", week: 2, actual: 20)
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                snapshotCalls += 1
                if snapshotCalls == 1 { return .init(status: 404, data: Data()) }
                let snapshot = SnapshotResponse(generatedAt: nil, weeks: [2], degradedPositions: [],
                                                scoring: ["ppr": [fresh], "standard": [fresh]])
                return .init(status: 200, data: try JSONEncoder().encode(snapshot))
            }
            let format = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
                .queryItems!.first { $0.name == "scoring" }!.value!
            let earlier = format == "standard"
            if earlier { started.fulfill() }
            let response = PredictionsResponse(players: [Self.player("old", week: 1, actual: 1)],
                                               total: 1, scoring: format, degradedPositions: [])
            return .init(status: earlier ? status : 200, data: try JSONEncoder().encode(response),
                         delay: earlier ? 0.5 : 0)
        }
        let store = SnapshotStore(api: client(), cache: SnapshotCache(directory: directory))
        await store.hydrate(scoring: .ppr)
        let earlier = Task { await store.ensureLive(.standard) }
        await fulfillment(of: [started], timeout: 2)
        await store.hydrate(scoring: .ppr)
        await earlier.value
        XCTAssertTrue(store.usingSnapshot)
        XCTAssertEqual(store.players(.ppr).first?.actual, 20)
        XCTAssertEqual(store.players(.standard).first?.actual, 20)
        XCTAssertEqual(store.weeks, [2])
        XCTAssertNil(store.errorMessage)
    }

    @MainActor
    func testLiveRefreshDiscardsOlderScoringResponse() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let started = expectation(description: "old scoring request started")
        var revision = 0
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                revision += 1
                return .init(status: 404, data: Data())
            }
            let format = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
                .queryItems!.first { $0.name == "scoring" }!.value!
            let earlier = format == "standard" && revision == 1
            if earlier { started.fulfill() }
            let response = PredictionsResponse(players: [Self.player(format, week: revision, actual: Double(revision))],
                                               total: 1, scoring: format, degradedPositions: [])
            return .init(status: 200, data: try JSONEncoder().encode(response), delay: earlier ? 0.5 : 0)
        }
        let store = SnapshotStore(api: client(), cache: SnapshotCache(directory: directory))
        await store.hydrate(scoring: .ppr)
        let earlier = Task { await store.ensureLive(.standard) }
        await fulfillment(of: [started], timeout: 2)
        await store.hydrate(scoring: .ppr)
        await earlier.value
        XCTAssertEqual(store.players(.ppr).first?.actual, 2)
        XCTAssertTrue(store.players(.standard).isEmpty)
        await store.ensureLive(.standard)
        XCTAssertEqual(store.players(.standard).first?.actual, 2)
    }

    @MainActor
    func testScoringChangeDuringSnapshotRefreshLoadsLatestFormat() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let started = expectation(description: "snapshot refresh started")
        var snapshotCalls = 0
        let old = Self.player("old", week: 1, actual: 1)
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                snapshotCalls += 1
                if snapshotCalls == 1 {
                    let snapshot = SnapshotResponse(generatedAt: nil, weeks: [1], degradedPositions: [],
                                                    scoring: ["ppr": [old], "standard": [old]])
                    return .init(status: 200, data: try JSONEncoder().encode(snapshot))
                }
                started.fulfill()
                return .init(status: 404, data: Data(), delay: 0.5)
            }
            let format = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
                .queryItems!.first { $0.name == "scoring" }!.value!
            let response = PredictionsResponse(players: [Self.player(format, week: 2, actual: 2)],
                                               total: 1, scoring: format, degradedPositions: [])
            return .init(status: 200, data: try JSONEncoder().encode(response))
        }
        let store = SnapshotStore(api: client(), cache: SnapshotCache(directory: directory))
        await store.hydrate(scoring: .ppr)
        let refreshing = Task { await store.hydrate(scoring: .ppr) }
        await fulfillment(of: [started], timeout: 2)
        // Mirrors LeadersView's scoring-change callback.
        if !store.usingSnapshot { await store.ensureLive(.standard) }
        await refreshing.value
        XCTAssertEqual(store.players(.standard).first?.actual, 2)
        XCTAssertFalse(store.usingSnapshot)
    }

    @MainActor
    func testEarlierHydrationFailureDoesNotEndCurrentLoading() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let firstStarted = expectation(description: "first hydration started")
        let secondStarted = expectation(description: "second hydration started")
        var calls = 0
        let fresh = Self.player("fresh", week: 2, actual: 2)
        FixtureURLProtocol.handler = { _ in
            calls += 1
            if calls == 1 {
                firstStarted.fulfill()
                return .init(status: 500, data: Data(), delay: 0.2)
            }
            secondStarted.fulfill()
            let snapshot = SnapshotResponse(generatedAt: nil, weeks: [2], degradedPositions: [], scoring: ["ppr": [fresh]])
            return .init(status: 200, data: try JSONEncoder().encode(snapshot), delay: 0.6)
        }
        let store = SnapshotStore(api: client(), cache: SnapshotCache(directory: directory))
        let earlier = Task { await store.hydrate(scoring: .ppr) }
        await fulfillment(of: [firstStarted], timeout: 2)
        let current = Task { await store.hydrate(scoring: .ppr) }
        await fulfillment(of: [secondStarted], timeout: 2)
        await earlier.value
        XCTAssertTrue(store.isLoading)
        XCTAssertNil(store.errorMessage)
        await current.value
        XCTAssertEqual(store.players(.ppr).first?.actual, 2)
    }

    @MainActor
    func testFailedRefreshKeepsNewerLiveDataOverOldDiskSnapshot() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let cache = SnapshotCache(directory: directory)
        let old = Self.player("old disk", week: 1, actual: 1)
        cache.save(try JSONEncoder().encode(SnapshotResponse(
            generatedAt: nil, weeks: [1], degradedPositions: [], scoring: ["ppr": [old]]
        )))
        var snapshots = 0
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                snapshots += 1
                return .init(status: snapshots == 1 ? 404 : 500, data: Data())
            }
            let response = PredictionsResponse(players: [Self.player("newer live", week: 2, actual: 2)],
                                               total: 1, scoring: "ppr", degradedPositions: [])
            return .init(status: 200, data: try JSONEncoder().encode(response))
        }
        let store = SnapshotStore(api: client(), cache: cache)
        await store.hydrate(scoring: .ppr)
        await store.hydrate(scoring: .ppr)
        XCTAssertEqual(store.players(.ppr).first?.actual, 2)
        XCTAssertEqual(store.weeks, [2])
    }

    @MainActor
    func testFailedSnapshotRefreshDoesNotSuppressFreshLiveResponse() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let refreshing = expectation(description: "snapshot refresh started")
        let liveStarted = expectation(description: "fresh live request started")
        var snapshots = 0
        let old = Self.player("old", week: 1, actual: 1)
        FixtureURLProtocol.handler = { request in
            if request.url!.path == "/api/snapshot" {
                snapshots += 1
                if snapshots == 1 {
                    let snapshot = SnapshotResponse(generatedAt: nil, weeks: [1], degradedPositions: [],
                                                    scoring: ["ppr": [old], "standard": [old]])
                    return .init(status: 200, data: try JSONEncoder().encode(snapshot))
                }
                refreshing.fulfill()
                return .init(status: 500, data: Data(), delay: 0.2)
            }
            liveStarted.fulfill()
            let response = PredictionsResponse(players: [Self.player("fresh", week: 2, actual: 2)],
                                               total: 1, scoring: "standard", degradedPositions: [])
            return .init(status: 200, data: try JSONEncoder().encode(response), delay: 0.5)
        }
        let store = SnapshotStore(api: client(), cache: SnapshotCache(directory: directory))
        await store.hydrate(scoring: .ppr)
        let refresh = Task { await store.hydrate(scoring: .ppr) }
        await fulfillment(of: [refreshing], timeout: 2)
        let live = Task { await store.ensureLive(.standard) }
        await fulfillment(of: [liveStarted], timeout: 2)
        await refresh.value
        await live.value
        XCTAssertEqual(store.players(.standard).first?.actual, 2)
        XCTAssertFalse(store.usingSnapshot)
    }

    @MainActor
    func testOldPlayerRequestFailureCannotReplaceNewScoring() async throws {
        try await verifyPlayerRequestRace(earlierStatus: 500)
    }

    @MainActor
    private func verifyPlayerRequestRace(earlierStatus: Int) async throws {
        let firstStarted = expectation(description: "first request started")
        FixtureURLProtocol.handler = { request in
            let format = URLComponents(url: request.url!, resolvingAgainstBaseURL: false)!
                .queryItems!.first { $0.name == "scoring" }!.value!
            let earlier = format == "ppr"
            if earlier { firstStarted.fulfill() }
            let value = PlayerDetail(playerID: "00-test", name: "Player", position: "QB", team: "KC",
                                     headshot: "", weekly: [], seasonAvg: 10, seasonTotal: 10, scoring: format)
            return .init(status: earlier ? earlierStatus : 200,
                         data: try JSONEncoder().encode(value), delay: earlier ? 0.5 : 0)
        }
        let store = PlayerDetailStore(api: client())
        let first = Task { await store.load(playerID: "00-test", week: nil, scoring: .ppr) }
        await fulfillment(of: [firstStarted], timeout: 2)
        await store.load(playerID: "00-test", week: nil, scoring: .standard)
        await first.value
        XCTAssertEqual(store.detail.value?.scoring, "standard")
        XCTAssertNil(store.detail.errorMessage)
    }

    func testRepeatedBenchmarkCommitKeepsDistinctRunIdentity() throws {
        let rows = """
        [{"timestamp":"2026-09-10T01:00:00","git_hash":"abc1234","pr_number":1,
          "training_skipped":false,"positions":[],"ridge":[],"nn":[],"attn_nn":[],"lgbm":[]},
         {"timestamp":"2026-09-10T02:00:00","git_hash":"abc1234","pr_number":1,
          "training_skipped":false,"positions":[],"ridge":[],"nn":[],"attn_nn":[],"lgbm":[]}]
        """
        let decoded = try JSONDecoder().decode([BenchmarkHistory.Row].self, from: Data(rows.utf8))
        XCTAssertNotEqual(decoded[0].id, decoded[1].id)
    }

    func testVegasSpreadUsesTheAPITeamMarginConvention() {
        // A +3.5 API margin implies (47.5 + 3.5) / 2 = 25.5 points for
        // this team versus 22 for its opponent: it is the favorite at -3.5.
        XCTAssertEqual(Fmt.vegasSpread(3.5).label, "Fav")
        XCTAssertEqual(Fmt.vegasSpread(3.5).value, "-3.5")
        XCTAssertEqual(Fmt.vegasSpread(-3.5).label, "Dog")
        XCTAssertEqual(Fmt.vegasSpread(-3.5).value, "+3.5")
    }
}
