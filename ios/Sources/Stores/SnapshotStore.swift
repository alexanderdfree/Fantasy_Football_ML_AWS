import Observation
import Foundation

/// Season Leaders data source. Snapshot-first: paint from the disk cache, fetch
/// `/api/snapshot` (all three scoring formats), and do all filtering/sorting
/// client-side (ports app.js). Falls back to live `/api/predictions` per scoring
/// when the snapshot is absent (404).
@MainActor
@Observable
final class SnapshotStore {
    private let api: any APIProviding
    private let cache: any SnapshotCaching
    private let decoder = JSONDecoder()

    var isLoading = false
    var errorMessage: String?
    var usingSnapshot = false
    var isStale = false
    var weeks: [Int] = []
    var teams: [String] = []
    var degradedPositions: [String] = []

    private var snapshot: SnapshotResponse?
    private var liveCache: [String: [Player]] = [:]
    private var loadGeneration = 0
    private var liveModeGeneration: Int?
    private var liveCacheGeneration: Int?
    private var liveRequestSequence = 0
    private var latestLiveRequests: [String: Int] = [:]
    private var latestScoring: ScoringFormat = .ppr

    init(api: any APIProviding = APIClient.shared, cache: any SnapshotCaching = SnapshotCache()) {
        self.api = api
        self.cache = cache
    }

    var hasData: Bool { snapshot != nil || !liveCache.isEmpty }

    func players(_ scoring: ScoringFormat) -> [Player] {
        snapshot?.players(scoring) ?? liveCache[scoring.rawValue] ?? []
    }

    func hydrate(scoring: ScoringFormat) async {
        loadGeneration += 1
        let generation = loadGeneration
        latestScoring = scoring
        defer { if generation == loadGeneration { isLoading = false } }
        if !hasData { isLoading = true }
        if !hasData, let cached = cache.load() {
            applySnapshot(cached)
            usingSnapshot = true
            isStale = true
        }
        do {
            let data = try await api.rawData(.snapshot)
            guard generation == loadGeneration else { return }
            let snap = try decoder.decode(SnapshotResponse.self, from: data)
            cache.save(data)
            liveModeGeneration = nil
            usingSnapshot = true
            applySnapshot(snap)
            isStale = false
            errorMessage = nil
        } catch let error as APIError where error.isNotFound {
            guard generation == loadGeneration else { return }
            liveModeGeneration = generation
            await loadLive(latestScoring, generation: generation)
        } catch {
            guard generation == loadGeneration else { return }
            isStale = hasData
            errorMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription
            if snapshot == nil, liveModeGeneration != nil {
                liveModeGeneration = generation
                await loadLive(latestScoring, generation: generation)
            }
        }
    }

    /// Live-mode only: ensure the active scoring format's rows are loaded
    /// (snapshot mode already holds all three).
    func ensureLive(_ scoring: ScoringFormat) async {
        latestScoring = scoring
        guard liveModeGeneration == loadGeneration,
              liveCacheGeneration != loadGeneration || liveCache[scoring.rawValue] == nil else { return }
        await loadLive(scoring, generation: loadGeneration)
    }

    private func applySnapshot(_ snap: SnapshotResponse) {
        snapshot = snap
        weeks = snap.weeks
        teams = snap.teams
        degradedPositions = snap.degradedPositions
    }

    private func loadLive(_ scoring: ScoringFormat, generation: Int) async {
        guard generation == loadGeneration, liveModeGeneration == generation else { return }
        liveRequestSequence += 1
        let request = liveRequestSequence
        latestLiveRequests[scoring.rawValue] = request
        do {
            let resp = try await api.get(
                .predictions(position: "ALL", week: "ALL", search: "", sort: "actual", order: "desc", scoring: scoring),
                as: PredictionsResponse.self
            )
            guard generation == loadGeneration, liveModeGeneration == generation,
                  scoring == latestScoring,
                  latestLiveRequests[scoring.rawValue] == request else { return }
            if liveCacheGeneration != generation {
                liveCache.removeAll()
                liveCacheGeneration = generation
            }
            liveCache[scoring.rawValue] = resp.players
            snapshot = nil
            usingSnapshot = false
            isStale = false
            degradedPositions = resp.degradedPositions
            weeks = Set(resp.players.map(\.week)).sorted()
            teams = Set(resp.players.map(\.team).filter { !$0.isEmpty }).sorted()
            errorMessage = nil
        } catch {
            guard generation == loadGeneration, liveModeGeneration == generation,
                  scoring == latestScoring,
                  latestLiveRequests[scoring.rawValue] == request else { return }
            isStale = hasData
            errorMessage = (error as? APIError)?.errorDescription ?? error.localizedDescription
        }
    }
}
